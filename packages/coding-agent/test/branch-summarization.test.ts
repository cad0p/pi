import type { AgentMessage, AgentTool, StreamFn } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	fauxToolCall,
	type Message,
	type Model,
	type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	buildStructuredSummaryMessages,
	generateBranchSummary,
	prepareBranchEntries,
	SUMMARIZATION_SYSTEM_PROMPT,
	stripBoundaryOrphanToolResults,
} from "../src/core/compaction/index.ts";
import type { SessionEntry } from "../src/core/session-manager.ts";

const model: Model<"anthropic-messages"> = {
	id: "test-model",
	name: "Test Model",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 200000,
	maxTokens: 8192,
};

const entries: SessionEntry[] = [
	{
		type: "message",
		id: "branch-user",
		parentId: null,
		timestamp: new Date(1).toISOString(),
		message: { role: "user", content: "Abandoned request", timestamp: 1 },
	},
];

function response(content: AssistantMessage["content"]): AssistantMessage {
	return {
		...fauxAssistantMessage(""),
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
	};
}

describe("branch summarization", () => {
	it("does not override tool choice for branch summaries", async () => {
		let requestOptions: SimpleStreamOptions | undefined;
		const streamFn: StreamFn = (_model, _context, options) => {
			requestOptions = options;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({ type: "done", reason: "stop", message: response([{ type: "text", text: "summary" }]) }),
			);
			return stream;
		};

		await generateBranchSummary(entries, {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(requestOptions?.maxTokens).toBe(4096);
		expect(requestOptions?.toolChoice).toBeUndefined();
	});

	it("clamps the branch summary output cap to the model limit", async () => {
		let requestOptions: SimpleStreamOptions | undefined;
		const streamFn: StreamFn = (_model, _context, options) => {
			requestOptions = options;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({ type: "done", reason: "stop", message: response([{ type: "text", text: "summary" }]) }),
			);
			return stream;
		};

		await generateBranchSummary(entries, {
			model: { ...model, maxTokens: 1024 },
			signal: new AbortController().signal,
			streamFn,
		});

		expect(requestOptions?.maxTokens).toBe(1024);
	});

	it("rejects tool calls from branch summaries", async () => {
		const streamFn: StreamFn = () => {
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({
					type: "done",
					reason: "toolUse",
					message: response([
						{ type: "toolCall", id: "tool-call-1", name: "read", arguments: { path: "README.md" } },
					]),
				}),
			);
			return stream;
		};

		const result = await generateBranchSummary(entries, {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(result.error).toBe("Branch summarization attempted to call a tool");
	});

	it("rejects length-limited branch summaries", async () => {
		const streamFn: StreamFn = () => {
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({
					type: "done",
					reason: "length",
					message: { ...response([{ type: "text", text: "partial" }]), stopReason: "length" },
				}),
			);
			return stream;
		};

		const result = await generateBranchSummary(entries, {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(result.error).toBe(
			"Branch summarization failed: generation hit the token cap and the summary is incomplete",
		);
	});
});

describe("cache-preserving branch summary request", () => {
	function toolResultMessage(toolCallId: string, text: string): AgentMessage {
		return {
			role: "toolResult",
			toolCallId,
			toolName: "read",
			content: [{ type: "text", text }],
			isError: false,
			timestamp: 3,
		};
	}
	const toolCallEntry: SessionEntry = {
		type: "message",
		id: "branch-tool-call",
		parentId: null,
		timestamp: new Date(2).toISOString(),
		message: fauxAssistantMessage([fauxToolCall("read", { path: "README.md" }, { id: "call-1" })], {
			stopReason: "toolUse",
			timestamp: 2,
		}),
	};
	const toolResultEntry: SessionEntry = {
		type: "message",
		id: "branch-tool-result",
		parentId: null,
		timestamp: new Date(3).toISOString(),
		message: toolResultMessage("call-1", "file contents"),
	};
	const orphanResultEntry: SessionEntry = {
		type: "message",
		id: "branch-orphan-result",
		parentId: null,
		timestamp: new Date(4).toISOString(),
		message: toolResultMessage("call-missing", "orphan"),
	};

	it("drops only tool results without a matching tool call", () => {
		const messages: Message[] = [
			fauxAssistantMessage([fauxToolCall("read", {}, { id: "call-1" })], { stopReason: "toolUse" }),
			toolResultMessage("call-1", "kept") as Message,
			toolResultMessage("call-missing", "dropped") as Message,
		];
		const stripped = stripBoundaryOrphanToolResults(messages);
		expect(stripped).toHaveLength(2);
		expect(stripped[1]).toMatchObject({ role: "toolResult", toolCallId: "call-1" });
		expect(messages).toHaveLength(3);
	});

	it("skips tool results by default but keeps them on request", () => {
		const legacy = prepareBranchEntries([toolCallEntry, toolResultEntry]);
		expect(legacy.messages.some((m) => m.role === "toolResult")).toBe(false);
		const live = prepareBranchEntries([toolCallEntry, toolResultEntry], 0, { includeToolResults: true });
		expect(live.messages.some((m) => m.role === "toolResult")).toBe(true);
		expect(live.messages).toHaveLength(2);
	});

	it("appends the instruction without touching the evidence prefix", () => {
		const evidence: Message[] = [
			{ role: "user", content: "hello", timestamp: 1 },
			{
				role: "toolResult",
				toolCallId: "call-missing",
				toolName: "read",
				content: [{ type: "text", text: "orphan" }],
				isError: false,
				timestamp: 2,
			},
		];
		const built = buildStructuredSummaryMessages(evidence, "do the thing", 1);
		expect(built).toHaveLength(2);
		expect(JSON.stringify(built.slice(0, 1))).toBe(JSON.stringify([evidence[0]]));
		const trailer = built[1];
		expect(trailer.role).toBe("user");
		expect(JSON.stringify(trailer).includes("<conversation>")).toBe(false);
	});

	it("numbers the first branch message after background entries", () => {
		const bg: SessionEntry = {
			type: "message",
			id: "bg",
			parentId: null,
			timestamp: new Date(1).toISOString(),
			message: { role: "user", content: "Background", timestamp: 1 },
		};
		const first: SessionEntry = {
			type: "message",
			id: "first",
			parentId: "bg",
			timestamp: new Date(2).toISOString(),
			message: { role: "user", content: "Branch start", timestamp: 2 },
		};
		const all = [bg, first];
		expect(prepareBranchEntries(all, 0).firstMessageNumber).toBe(1);
		expect(prepareBranchEntries(all, 0, { branchStartId: first.id }).firstMessageNumber).toBe(2);
	});

	it("substitutes the strip-adjusted branch number into the instruction", () => {
		const evidence: Message[] = [
			{
				role: "toolResult",
				toolCallId: "call-missing",
				toolName: "read",
				content: [{ type: "text", text: "orphan" }],
				isError: false,
				timestamp: 1,
			},
			{ role: "user", content: "hello", timestamp: 2 },
		];
		const built = buildStructuredSummaryMessages(evidence, "Summarize only messages {first} onwards", 2);
		expect(built).toHaveLength(2);
		expect(JSON.stringify(built[1])).toContain("messages 1 onwards");
	});

	it("sends structured history under the live prefix when requestContext is provided", async () => {
		let requestContext: { systemPrompt?: string; messages?: unknown; tools?: unknown } | undefined;
		let requestOptions: SimpleStreamOptions | undefined;
		const streamFn: StreamFn = (_model, context, options) => {
			requestContext = {
				systemPrompt: (context as { systemPrompt?: string }).systemPrompt,
				messages: (context as { messages?: unknown }).messages,
				tools: (context as { tools?: unknown }).tools,
			};
			requestOptions = options;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({ type: "done", reason: "stop", message: response([{ type: "text", text: "summary" }]) }),
			);
			return stream;
		};
		const tools = [{ name: "read" } as AgentTool<any>];

		const result = await generateBranchSummary([toolCallEntry, toolResultEntry, orphanResultEntry], {
			model,
			signal: new AbortController().signal,
			streamFn,
			requestContext: { systemPrompt: "live system", tools, sessionId: "session-1" },
			thinkingLevel: "xhigh",
		});

		expect(result.error).toBeUndefined();
		expect(requestContext?.systemPrompt).toBe("live system");
		expect(requestContext?.tools).toBe(tools);
		const sent = requestContext?.messages as Message[];
		expect(sent).toHaveLength(3);
		expect(sent[2].role).toBe("user");
		expect(JSON.stringify(sent).includes("<conversation>")).toBe(false);
		expect(sent.some((m) => m.role === "toolResult" && m.toolCallId === "call-missing")).toBe(false);
		expect(requestOptions?.sessionId).toBe("session-1");
		expect(requestOptions?.cacheRetention).toBe("short");
		expect(requestOptions?.reasoning).toBe("xhigh");
		expect(requestOptions?.maxTokens).toBe(4096);
	});

	it("keeps the legacy standalone request without requestContext", async () => {
		let requestContext: { systemPrompt?: string; messages?: unknown; tools?: unknown } | undefined;
		let requestOptions: SimpleStreamOptions | undefined;
		const streamFn: StreamFn = (_model, context, options) => {
			requestContext = {
				systemPrompt: (context as { systemPrompt?: string }).systemPrompt,
				messages: (context as { messages?: unknown }).messages,
				tools: (context as { tools?: unknown }).tools,
			};
			requestOptions = options;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({ type: "done", reason: "stop", message: response([{ type: "text", text: "summary" }]) }),
			);
			return stream;
		};

		await generateBranchSummary(entries, {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(requestContext?.systemPrompt).toBe(SUMMARIZATION_SYSTEM_PROMPT);
		expect(requestContext?.tools).toBeUndefined();
		const sent = requestContext?.messages as Message[];
		expect(sent).toHaveLength(1);
		expect(JSON.stringify(sent[0]).includes("<conversation>")).toBe(true);
		expect(requestOptions?.cacheRetention).toBe("none");
		expect(requestOptions?.reasoning).toBeUndefined();
	});
});
