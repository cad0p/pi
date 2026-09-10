/**
 * Cache-miss notice wiring for branch summaries (classic copy).
 *
 * The summary request reuses the live prompt-cache prefix; when the measured
 * response usage shows the prefix was not read back from cache, the miss is
 * persisted on the branch_summary entry so rebuilds re-render the notice.
 * Warm hits stay silent, and extension-provided summaries (usage not measured
 * here, model unknown) never record a miss.
 *
 * Detection assertions target the production seam (detectBranchSummaryCacheMiss)
 * and the real AgentSession.navigateTree wiring. A no-cache provider cannot be
 * driven through navigateTree — the faux provider always simulates cache
 * activity — so that shape is covered at the seam, where the guard lives.
 */

import type { BranchSummaryCacheMiss as HarnessBranchSummaryCacheMiss, StreamFn } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, fauxAssistantMessage } from "@earendil-works/pi-ai";
import type { AssistantMessage, Usage } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it } from "vitest";
import {
	type CacheMiss,
	collectCacheMisses,
	computeCacheWaste,
	detectBranchSummaryCacheMiss,
	type ModelPriceSource,
} from "../src/core/cache-stats.ts";
import { generateBranchSummary } from "../src/core/compaction/index.ts";
import { type SessionEntry, SessionManager } from "../src/core/session-manager.ts";
import { getUsageCostBreakdown } from "../src/core/usage-totals.ts";
import { createHarness, type Harness } from "./suite/harness.ts";
import { userMsg } from "./utilities.ts";

// Compile-time drift guard: the harness miss shape must stay identical to the
// classic one (separate packages, so they cannot share the type).
type MutuallyAssignable<A, B> = A extends B ? (B extends A ? true : never) : never;
const _missShapeParity: MutuallyAssignable<CacheMiss, HarnessBranchSummaryCacheMiss> = true;

const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };

function usage(overrides: Partial<Usage>): Usage {
	return {
		input: 0,
		output: 50,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 50,
		cost: { ...zeroCost },
		...overrides,
	};
}

const models: ModelPriceSource = {
	// $/million tokens; used as cache-read price fallback on full-miss turns
	getModel: () => ({ cost: { cacheRead: 0.3 } }),
};

function assistantEntry(id: string, parentId: string | null, messageUsage: Usage, timestamp = 0): SessionEntry {
	const message: AssistantMessage = {
		role: "assistant",
		content: [{ type: "text", text: "prior reply" }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "test-model",
		usage: messageUsage,
		stopReason: "stop",
		timestamp,
	};
	return { type: "message", id, parentId, timestamp: new Date(timestamp).toISOString(), message };
}

// Prior turn: 100k prompt cached (write reported), so the next request should
// read it back.
const warmedEntries: SessionEntry[] = [assistantEntry("turn-1", null, usage({ cacheWrite: 100_000 }), 0)];

function summaryEntry(id: string, entryUsage: Usage, timestamp = 1): SessionEntry {
	return {
		type: "branch_summary",
		id,
		parentId: null,
		timestamp: new Date(timestamp).toISOString(),
		fromId: "branch-tip",
		summary: "earlier summary",
		usage: entryUsage,
	};
}

function probeUsage(responseUsage: Usage): { usage: Usage; provider: string; model: string; timestamp: number } {
	return { usage: responseUsage, provider: "anthropic", model: "test-model", timestamp: 60_000 };
}

function detect(entries: SessionEntry[], responseUsage: Usage): CacheMiss | undefined {
	const probe = probeUsage(responseUsage);
	return detectBranchSummaryCacheMiss(entries, probe.usage, probe.provider, probe.model, probe.timestamp, models);
}

describe("branch summary cache-miss detection", () => {
	it("measures a miss on a cold response", () => {
		expect(detect(warmedEntries, usage({ input: 100_000 }))?.missedTokens).toBe(100_000);
	});

	it("stays silent on a warm response", () => {
		expect(detect(warmedEntries, usage({ input: 2_000, cacheRead: 100_000 }))).toBeUndefined();
	});

	it.each([
		{ cacheRead: 100_000 - 1_024, expected: undefined },
		{ cacheRead: 100_000 - 1_025, expected: 1_025 },
	])("noise boundary: cacheRead $cacheRead", ({ cacheRead, expected }) => {
		const miss = detect(warmedEntries, usage({ input: 100_000, cacheRead }));
		if (expected === undefined) expect(miss).toBeUndefined();
		else expect(miss?.missedTokens).toBe(expected);
	});

	it("stays silent with no previous request", () => {
		expect(detect([], usage({ input: 100_000 }))).toBeUndefined();
	});

	it("keeps session cache capability across a summary boundary (E2E: earlier summary hit, later full miss)", () => {
		const history = [
			assistantEntry("a", null, usage({ cacheWrite: 100_000 }), 0),
			summaryEntry("s", usage({ input: 500, cacheRead: 61_425 }), 1),
			assistantEntry("b", "s", usage({ input: 2_000 }), 2),
		];
		// The earlier summary's cache read proves the provider reports caching,
		// so the within-segment zero-read still counts despite the reset.
		expect(detect(history, usage({ input: 63_000 }))?.missedTokens).toBe(2_000);
	});

	it("stays silent on a warm second summary (parent baseline read back)", () => {
		const history = [
			assistantEntry("a", null, usage({ cacheWrite: 100_000 }), 0),
			summaryEntry("s", usage({ input: 500, cacheRead: 61_425 }), 1),
			assistantEntry("b", "s", usage({ input: 2_000 }), 2),
		];
		// Warm: the probe reads back the 2k parent baseline.
		expect(detect(history, usage({ input: 2_000, cacheRead: 61_000 }))).toBeUndefined();
	});

	it("stays silent-first after a compaction boundary (baseline invalid)", () => {
		const compaction = { type: "compaction", id: "c", parentId: null, timestamp: "" } as SessionEntry;
		const history = [assistantEntry("a", null, usage({ cacheWrite: 100_000 }), 0), compaction];
		// Compaction rewrites the prompt: the cold probe is new content, not a miss.
		expect(detect(history, usage({ input: 63_000 }))).toBeUndefined();
	});

	it("still skips cache-less providers across a summary boundary", () => {
		const history = [
			assistantEntry("a", null, usage({ input: 100_000 }), 0),
			summaryEntry("s", usage({ input: 500 }), 1),
			assistantEntry("b", "s", usage({ input: 2_000 }), 2),
		];
		expect(detect(history, usage({ input: 63_000 }))).toBeUndefined();
	});

	it("skips providers that report no cache activity", () => {
		const history = [
			assistantEntry("a", null, usage({ input: 100_000 }), 0),
			assistantEntry("b", "a", usage({ input: 110_000 }), 1),
		];
		expect(detect(history, usage({ input: 110_000 }))).toBeUndefined();
	});

	it("stays silent on a warm response after a model switch", () => {
		const probe = probeUsage(usage({ input: 2_000, cacheRead: 100_000 }));
		const miss = detectBranchSummaryCacheMiss(
			warmedEntries,
			probe.usage,
			probe.provider,
			"other-model",
			probe.timestamp,
			models,
		);
		// Silent via provider-switch suppression (as well as the warm read-back).
		expect(miss).toBeUndefined();
	});

	it("stays silent on a cold response after a model switch (expected re-billing)", () => {
		const probe = probeUsage(usage({ input: 100_000 }));
		const miss = detectBranchSummaryCacheMiss(
			warmedEntries,
			probe.usage,
			probe.provider,
			"other-model",
			probe.timestamp,
			models,
		);
		// A new model cannot read the previous prefix: expected re-billing,
		// not an actionable miss.
		expect(miss).toBeUndefined();
	});

	it("ignores branch summaries in live-turn totals", () => {
		const history = [
			assistantEntry("a", null, usage({ cacheWrite: 100_000 }), 0),
			summaryEntry("s", usage({ input: 500, cacheRead: 61_425 }), 1),
			assistantEntry("b", "s", usage({ input: 2_000 }), 2),
		];
		// Live accounting still resets at the summary boundary: the post-summary
		// turn is new content, and the summary usage itself is never a miss.
		expect(computeCacheWaste(history, models)).toMatchObject({
			missedTokens: 0,
			missCount: 0,
		});
		expect(collectCacheMisses(history, models).size).toBe(0);
	});

	it("measures the usage returned by the generator instead of the request shape", async () => {
		// Legacy standalone path (no requestContext, cacheRetention "none"):
		// a cold measured usage still counts — nothing is hardcoded per path.
		const streamFn: StreamFn = (_model, _context, _options) => {
			const stream = createAssistantMessageEventStream();
			const message: AssistantMessage = {
				role: "assistant",
				content: [{ type: "text", text: "## Goal\nA summary" }],
				api: "anthropic-messages",
				provider: "anthropic",
				model: "test-model",
				usage: usage({ input: 100_000 }),
				stopReason: "stop",
				timestamp: 999,
			};
			queueMicrotask(() => stream.push({ type: "done", reason: "stop", message }));
			return stream;
		};
		const branchEntries: SessionEntry[] = [
			{
				type: "message",
				id: "branch-user",
				parentId: null,
				timestamp: new Date(1).toISOString(),
				message: { role: "user", content: "Abandoned request", timestamp: 1 },
			},
		];
		const result = await generateBranchSummary(branchEntries, {
			model: {
				id: "test-model",
				name: "Test Model",
				api: "anthropic-messages",
				provider: "anthropic",
				baseUrl: "https://api.anthropic.com",
				reasoning: false,
				input: ["text"],
				cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
				contextWindow: 200000,
				maxTokens: 8192,
			},
			signal: new AbortController().signal,
			streamFn,
		});
		expect(result.error).toBeUndefined();
		expect(result.usage ? detect(warmedEntries, result.usage)?.missedTokens : undefined).toBe(100_000);
	});
});

describe("branch summary cache-miss persistence", () => {
	it("persists the measured miss without changing billed-token accounting", () => {
		const manager = SessionManager.inMemory();
		const summaryUsage = usage({ input: 10, output: 20, cacheRead: 30, cacheWrite: 40 });
		summaryUsage.cost = { input: 0.1, output: 0.2, cacheRead: 0.3, cacheWrite: 0.4, total: 1 };
		const cacheMiss: CacheMiss = { missedTokens: 50_000, missedCost: 0.15, idleMs: 1_000, modelChanged: false };

		const id = manager.branchWithSummary(null, "summary", undefined, false, summaryUsage, cacheMiss);
		expect(manager.getEntry(id)).toMatchObject({ type: "branch_summary", cacheMiss });

		// Display metadata only: cost breakdown tokens match the usage alone.
		const breakdown = getUsageCostBreakdown(manager.getEntries());
		expect(breakdown).toHaveLength(1);
		expect(breakdown[0]).toMatchObject({
			key: "Tools/summaries",
			tokens: 10 + 20 + 30 + 40,
			cost: 1,
		});
	});

	it("leaves entries without a miss unchanged", () => {
		const manager = SessionManager.inMemory();
		const id = manager.branchWithSummary(null, "summary", undefined, false, usage({ input: 5 }));
		expect(manager.getEntry(id)).toMatchObject({ type: "branch_summary" });
		expect((manager.getEntry(id) as { cacheMiss?: unknown }).cacheMiss).toBeUndefined();
	});
});

describe("branch summary cache-miss navigation wiring", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	function bigAssistant(text: string, provider: string, modelId: string): AssistantMessage {
		return {
			role: "assistant",
			content: [{ type: "text", text }],
			api: "anthropic-messages",
			provider,
			model: modelId,
			usage: usage({ output: 20, cacheWrite: 100_000 }),
			stopReason: "stop",
			timestamp: Date.now(),
		};
	}

	async function seedWarmedBranch(harness: Harness): Promise<string> {
		// Seed history on the same model the summarizer runs, so a measured
		// miss is not confounded by a model switch.
		const seedModel = harness.getModel();
		const targetId = harness.sessionManager.appendMessage(userMsg("keep this"));
		harness.sessionManager.appendMessage(bigAssistant("warmed prefix reply", seedModel.provider, seedModel.id));
		harness.sessionManager.appendMessage(userMsg(`abandoned work ${"x".repeat(8_000)}`));
		harness.sessionManager.appendMessage(bigAssistant("abandoned reply", seedModel.provider, seedModel.id));
		return targetId;
	}

	it("persists the measured miss on navigateTree without changing session totals", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		const targetId = await seedWarmedBranch(harness);
		harness.setResponses([fauxAssistantMessage("## Goal\nabandoned summary")]);

		const before = harness.session.getSessionStats();
		const result = await harness.session.navigateTree(targetId, { summarize: true });
		const summaryEntry = result.summaryEntry;

		expect(summaryEntry?.type).toBe("branch_summary");
		expect(summaryEntry?.fromHook).not.toBe(true);
		expect(summaryEntry?.usage).toBeDefined();
		// First summary request on this session id is cold: nothing was read back.
		expect(summaryEntry?.cacheMiss).toBeDefined();
		expect(summaryEntry?.cacheMiss?.missedTokens).toBeGreaterThan(1_024);
		expect(summaryEntry?.cacheMiss?.modelChanged).toBe(false);

		// Totals-neutral: session stat deltas equal the summary usage buckets exactly.
		const after = harness.session.getSessionStats();
		const summaryUsage = summaryEntry?.usage as Usage;
		for (const bucket of ["input", "output", "cacheRead", "cacheWrite"] as const) {
			expect(after.tokens[bucket] - before.tokens[bucket]).toBe(summaryUsage[bucket]);
		}
		expect(after.cost - before.cost).toBeCloseTo(summaryUsage.cost.total, 8);
	});

	it("stays silent on navigateTree with no previous request", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		// User-only branch: nothing to miss against.
		const targetId = harness.sessionManager.appendMessage(userMsg("keep this"));
		harness.sessionManager.appendMessage(userMsg("abandoned question one"));
		harness.sessionManager.appendMessage(userMsg("abandoned question two"));
		harness.setResponses([fauxAssistantMessage("## Goal\nabandoned summary")]);

		const result = await harness.session.navigateTree(targetId, { summarize: true });
		expect(result.summaryEntry?.type).toBe("branch_summary");
		expect(result.summaryEntry?.usage).toBeDefined();
		expect(result.summaryEntry?.cacheMiss).toBeUndefined();
	});

	it("still measures a miss when the prefix is truncated over budget", async () => {
		const harness = await createHarness({
			models: [{ id: "small-model", contextWindow: 20_000 }],
			settings: { branchSummary: { reserveTokens: 15_000 } },
		});
		harnesses.push(harness);
		const seedModel = harness.getModel();
		// 5k-token budget: the newest 4k-token message fits, the older one drops.
		const targetId = harness.sessionManager.appendMessage(userMsg("keep this"));
		harness.sessionManager.appendMessage(bigAssistant("warmed prefix reply", seedModel.provider, seedModel.id));
		harness.sessionManager.appendMessage(userMsg(`older work ${"x".repeat(16_000)}`));
		harness.sessionManager.appendMessage(userMsg(`newer work ${"x".repeat(16_000)}`));
		harness.setResponses([fauxAssistantMessage("## Goal\ntruncated summary")]);

		const result = await harness.session.navigateTree(targetId, { summarize: true });
		expect(result.summaryEntry?.type).toBe("branch_summary");
		expect(result.summaryEntry?.usage).toBeDefined();
		expect(result.summaryEntry?.cacheMiss?.missedTokens).toBeGreaterThan(1_024);
	});

	it("stays silent on a model switch measured on navigateTree", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		// Seed history on a different model than the faux summarizer runs.
		const targetId = harness.sessionManager.appendMessage(userMsg("keep this"));
		harness.sessionManager.appendMessage(bigAssistant("warmed prefix reply", "anthropic", "other-model"));
		harness.sessionManager.appendMessage(userMsg(`abandoned work ${"x".repeat(8_000)}`));
		harness.sessionManager.appendMessage(bigAssistant("abandoned reply", "anthropic", "other-model"));
		harness.setResponses([fauxAssistantMessage("## Goal\nabandoned summary")]);

		const result = await harness.session.navigateTree(targetId, { summarize: true });
		// A new model cannot read the previous prefix: expected re-billing,
		// not an actionable miss.
		expect(result.summaryEntry?.cacheMiss).toBeUndefined();
	});

	it("records no miss for extension-provided summaries", async () => {
		const summaryUsage = usage({ input: 10, output: 20, cacheRead: 30, cacheWrite: 40 });
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.on("session_before_tree", () => ({
						summary: { summary: "Summary provided by extension", usage: summaryUsage },
					}));
				},
			],
		});
		harnesses.push(harness);

		const targetId = harness.sessionManager.appendMessage(userMsg("first branch"));
		harness.sessionManager.appendMessage(bigAssistant("first reply", "anthropic", "test"));
		harness.sessionManager.appendMessage(userMsg("abandoned branch work"));
		harness.sessionManager.appendMessage(bigAssistant("abandoned reply", "anthropic", "test"));

		const result = await harness.session.navigateTree(targetId, { summarize: true });
		expect(result.summaryEntry?.type).toBe("branch_summary");
		expect(result.summaryEntry?.usage).toEqual(summaryUsage);
		expect(result.summaryEntry?.cacheMiss).toBeUndefined();
	});
});
