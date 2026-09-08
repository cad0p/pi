import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Message, Tool } from "@earendil-works/pi-ai";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	buildStructuredSummaryMessages,
	collectEntriesForBranchSummary,
	generateBranchSummaryWithRequest,
	prepareBranchEntries,
} from "../../src/harness/compaction/branch-summarization.ts";
import { SUMMARIZATION_SYSTEM_PROMPT } from "../../src/harness/compaction/compaction.ts";
import { stripBoundaryOrphanToolResults } from "../../src/harness/compaction/utils.ts";
import { BACKGROUND_CONTEXT } from "../../src/harness/context.ts";
import type { Branch, Entry, MessageEntry, Session } from "../../src/harness/session/index.ts";

function message(text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: 1 };
}

function messageEntry(id: string, parentId: string | null, text: string, seq: number): MessageEntry {
	return { type: "message", id, parentId, message: message(text), seq, timestamp: seq };
}

function branchReader(entries: Entry[]): {
	branch: Pick<Branch, "findEntries">;
	session: Pick<Session, "getEntry">;
} {
	const byId = new Map(entries.map((entry) => [entry.id, entry]));
	return {
		session: {
			async getEntry(id) {
				return byId.get(id);
			},
		},
		branch: {
			async findEntries(query = {}) {
				const path: Entry[] = [];
				let currentId = query.start ?? null;
				while (currentId !== null) {
					const entry = byId.get(currentId);
					if (!entry) throw new Error(`Unknown entry ${currentId}`);
					path.push(entry);
					currentId = entry.parentId;
				}
				return path;
			},
		},
	};
}

describe("v4 branch summarization", () => {
	it("collects the abandoned side of a branch in chronological order", async () => {
		const root = messageEntry("root", null, "root", 1);
		const common = messageEntry("common", root.id, "common", 2);
		const abandoned1 = messageEntry("abandoned-1", common.id, "abandoned 1", 3);
		const abandoned2 = messageEntry("abandoned-2", abandoned1.id, "abandoned 2", 4);
		const target = messageEntry("target", common.id, "target", 5);
		const { branch, session } = branchReader([root, common, abandoned1, abandoned2, target]);

		const result = await collectEntriesForBranchSummary(
			branch,
			session,
			abandoned2.id,
			target.id,
			BACKGROUND_CONTEXT,
		);
		expect(result.commonAncestorId).toBe(common.id);
		expect(result.entries.map((entry) => entry.id)).toEqual([abandoned1.id, abandoned2.id]);
		expect(result.entries.some((entry) => entry.id === root.id)).toBe(false);
	});

	it("returns no entries when there was no previous leaf", async () => {
		const target = messageEntry("target", null, "target", 1);
		const { branch, session } = branchReader([target]);
		expect(await collectEntriesForBranchSummary(branch, session, null, target.id, BACKGROUND_CONTEXT)).toEqual({
			entries: [],
			commonAncestorId: null,
		});
	});
});

describe("cache-preserving branch summary request", () => {
	function assistantWithCall(id: string): AgentMessage {
		return fauxAssistantMessage([fauxToolCall("read", {}, { id })], { stopReason: "toolUse", timestamp: 1 });
	}
	function resultFor(id: string): AgentMessage {
		return {
			role: "toolResult",
			toolCallId: id,
			toolName: "read",
			content: [{ type: "text", text: "out" }],
			isError: false,
			timestamp: 2,
		};
	}

	it("drops only tool results without a matching tool call", () => {
		const messages: Message[] = [
			assistantWithCall("call-1") as Message,
			resultFor("call-1") as Message,
			resultFor("call-missing") as Message,
		];
		const stripped = stripBoundaryOrphanToolResults(messages);
		expect(stripped).toHaveLength(2);
		expect(stripped[1]).toMatchObject({ role: "toolResult", toolCallId: "call-1" });
	});

	it("appends the instruction without touching the evidence prefix", () => {
		const evidence: Message[] = [{ role: "user", content: "hello", timestamp: 1 } as Message];
		const built = buildStructuredSummaryMessages(evidence, "do the thing");
		expect(built).toHaveLength(2);
		expect(JSON.stringify(built.slice(0, 1))).toBe(JSON.stringify(evidence));
		expect(built[1].role).toBe("user");
		expect(JSON.stringify(built).includes("<conversation>")).toBe(false);
	});

	it("sends structured history under the live prefix when requestContext is provided", async () => {
		let seen: { systemPrompt?: string; messages?: unknown; tools?: unknown } | undefined;
		let seenOptions:
			| { cacheRetention?: unknown; sessionId?: unknown; maxTokens?: unknown; reasoning?: unknown }
			| undefined;
		const response: AssistantMessage = fauxAssistantMessage("summary", { timestamp: 1 });
		const tools = [{ name: "read", description: "r", parameters: {} } as Tool];
		const callEntry = {
			type: "message",
			id: "c",
			parentId: null,
			message: assistantWithCall("call-1"),
			seq: 1,
			timestamp: 1,
		} as const;
		const resultEntry = {
			type: "message",
			id: "r",
			parentId: "c",
			message: resultFor("call-1"),
			seq: 2,
			timestamp: 2,
		} as const;
		const preparation = prepareBranchEntries([callEntry, resultEntry], 0, { includeToolResults: true });
		expect(preparation.messages.some((m) => m.role === "toolResult")).toBe(true);

		const result = await generateBranchSummaryWithRequest(
			preparation,
			{
				thinkingLevel: "xhigh",
				requestContext: { systemPrompt: "live system", tools, sessionId: "lane-1" },
			},
			async (aiContext, options) => {
				seen = {
					systemPrompt: (aiContext as { systemPrompt?: string }).systemPrompt,
					messages: (aiContext as { messages?: unknown }).messages,
					tools: (aiContext as { tools?: unknown }).tools,
				};
				seenOptions = {
					cacheRetention: (options as { cacheRetention?: unknown }).cacheRetention,
					sessionId: (options as { sessionId?: unknown }).sessionId,
					maxTokens: (options as { maxTokens?: unknown }).maxTokens,
					reasoning: (options as { reasoning?: unknown }).reasoning,
				};
				return response;
			},
			BACKGROUND_CONTEXT,
		);

		expect(result.ok).toBe(true);
		expect(seen?.systemPrompt).toBe("live system");
		expect(seen?.tools).toBe(tools);
		const sent = seen?.messages as Message[];
		expect(sent).toHaveLength(3);
		expect(sent[2].role).toBe("user");
		expect(JSON.stringify(sent).includes("<conversation>")).toBe(false);
		expect(seenOptions?.sessionId).toBe("lane-1");
		expect(seenOptions?.cacheRetention).toBe("short");
		expect(seenOptions?.reasoning).toBe("xhigh");
		expect(seenOptions?.maxTokens).toBe(2048);
	});

	it("keeps the legacy standalone request without requestContext", async () => {
		let seen: { systemPrompt?: string; messages?: unknown; tools?: unknown } | undefined;
		let seenOptions: { cacheRetention?: unknown } | undefined;
		const response: AssistantMessage = fauxAssistantMessage("summary", { timestamp: 1 });
		const callEntry = {
			type: "message",
			id: "c",
			parentId: null,
			message: message("evidence"),
			seq: 1,
			timestamp: 1,
		} as const;
		const preparation = prepareBranchEntries([callEntry]);
		const result = await generateBranchSummaryWithRequest(
			preparation,
			{},
			async (aiContext, options) => {
				seen = {
					systemPrompt: (aiContext as { systemPrompt?: string }).systemPrompt,
					messages: (aiContext as { messages?: unknown }).messages,
					tools: (aiContext as { tools?: unknown }).tools,
				};
				seenOptions = { cacheRetention: (options as { cacheRetention?: unknown }).cacheRetention };
				return response;
			},
			BACKGROUND_CONTEXT,
		);

		expect(result.ok).toBe(true);
		expect(seen?.systemPrompt).toBe(SUMMARIZATION_SYSTEM_PROMPT);
		expect(seen?.tools).toBeUndefined();
		expect((seen?.messages as Message[]).length).toBe(1);
		expect(seenOptions?.cacheRetention).toBe("none");
	});
});
