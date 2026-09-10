/**
 * Cache-miss measurement for branch-summary responses (harness copy).
 *
 * The summary request reuses the live lane prefix; when the measured response
 * usage shows the prefix was not read back from cache, the drive persists the
 * miss on the branch_summary entry. Warm hits stay silent. Same thresholds as
 * the classic scan; the legacy retention-none path is measured, never hardcoded.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Usage } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	detectBranchSummaryCacheMiss,
	type ModelPriceSource,
	shouldDisplayBranchSummaryCacheMiss,
} from "../../src/harness/compaction/cache-miss.ts";

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

function userMessage(text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: 1 };
}

function assistantMessage(responseUsage: Usage, options?: { timestamp?: number; model?: string }): AgentMessage {
	const message: AssistantMessage = {
		role: "assistant",
		content: [{ type: "text", text: "prior reply" }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: options?.model ?? "test-model",
		usage: responseUsage,
		stopReason: "stop",
		timestamp: options?.timestamp ?? 0,
	};
	return message;
}

function summaryMessage(summaryUsage: Usage): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "summary" }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "test-model",
		usage: summaryUsage,
		stopReason: "stop",
		timestamp: 60_000,
	};
}

// Prior turn: 100k prompt cached (write reported), so the summary should read it back.
const warmedHistory: AgentMessage[] = [userMessage("question"), assistantMessage(usage({ cacheWrite: 100_000 }))];

describe("detectBranchSummaryCacheMiss", () => {
	it("measures a miss on a cold response", () => {
		const miss = detectBranchSummaryCacheMiss(warmedHistory, summaryMessage(usage({ input: 100_000 })), models);
		expect(miss?.missedTokens).toBe(100_000);
		expect(miss?.missedCost).toBe(0);
		expect(miss?.idleMs).toBe(60_000);
		expect(miss?.modelChanged).toBe(false);
	});

	it("prices the miss from the paid and cache-read rates", () => {
		const responseUsage = usage({ input: 100_000 });
		responseUsage.cost = { ...zeroCost, input: 0.375, total: 0.375 };
		const miss = detectBranchSummaryCacheMiss(warmedHistory, summaryMessage(responseUsage), models);
		// 100k at ($3.75 - $0.30)/M
		expect(miss?.missedCost).toBeCloseTo(0.345, 5);
	});

	it("stays silent on a warm response", () => {
		expect(
			detectBranchSummaryCacheMiss(
				warmedHistory,
				summaryMessage(usage({ input: 2_000, cacheRead: 100_000 })),
				models,
			),
		).toBeUndefined();
	});

	it("stays silent with no previous request", () => {
		expect(
			detectBranchSummaryCacheMiss([userMessage("first")], summaryMessage(usage({ input: 100_000 })), models),
		).toBeUndefined();
	});

	it("ignores misses at or below the noise floor", () => {
		expect(
			detectBranchSummaryCacheMiss(warmedHistory, summaryMessage(usage({ input: 500 })), models),
		).toBeUndefined();
	});

	it("skips providers that report no cache activity", () => {
		const history: AgentMessage[] = [
			userMessage("a"),
			assistantMessage(usage({ input: 100_000 })),
			userMessage("b"),
			assistantMessage(usage({ input: 110_000 })),
		];
		expect(detectBranchSummaryCacheMiss(history, summaryMessage(usage({ input: 110_000 })), models)).toBeUndefined();
	});

	it("resets the scan at prior compaction summaries and measures from the newer segment", () => {
		const history: AgentMessage[] = [
			assistantMessage(usage({ cacheWrite: 100_000 })),
			{ role: "compactionSummary", summary: "compacted", tokensBefore: 1_000, timestamp: 2 },
			userMessage("newer work"),
			assistantMessage(usage({ cacheWrite: 50_000 }), { timestamp: 3 }),
		];
		const miss = detectBranchSummaryCacheMiss(
			history,
			{ ...summaryMessage(usage({ input: 60_000 })), timestamp: 4 },
			models,
		);
		// Previous prompt is the post-reset 50k turn, not the pre-reset 100k one.
		expect(miss?.missedTokens).toBe(50_000);
		expect(miss?.idleMs).toBe(1);
	});

	it("keeps the parent baseline across a branch summary (no reset)", () => {
		const history: AgentMessage[] = [
			assistantMessage(usage({ cacheWrite: 100_000 })),
			{ role: "branchSummary", summary: "older branch", fromId: "old", timestamp: 2 },
			userMessage("newer work"),
			assistantMessage(usage({ input: 2_000 }), { timestamp: 3 }),
		];
		// Branch summaries reuse the live prefix: the probe measures against the
		// 2k parent baseline, min(parent 2k, probe 60k).
		const miss = detectBranchSummaryCacheMiss(history, summaryMessage(usage({ input: 60_000 })), models);
		expect(miss?.missedTokens).toBe(2_000);
	});

	it("measures a miss on a consecutive summary without an intervening turn", () => {
		const history: AgentMessage[] = [
			assistantMessage(usage({ cacheWrite: 100_000 })),
			{ role: "branchSummary", summary: "older branch", fromId: "old", timestamp: 2 },
		];
		// No reset at branch summaries: the parent 100k baseline survives.
		const miss = detectBranchSummaryCacheMiss(history, summaryMessage(usage({ input: 60_000 })), models);
		expect(miss?.missedTokens).toBe(60_000);
	});

	it("stays silent-first after a compaction summary (baseline invalid)", () => {
		const history: AgentMessage[] = [
			assistantMessage(usage({ cacheWrite: 100_000 })),
			{ role: "compactionSummary", summary: "compacted", tokensBefore: 1_000, timestamp: 2 },
		];
		// Pre-compaction prompts are rewritten: the cold probe is new content.
		expect(detectBranchSummaryCacheMiss(history, summaryMessage(usage({ input: 60_000 })), models)).toBeUndefined();
	});

	it("stays silent on a warm response across a branch summary", () => {
		const history: AgentMessage[] = [
			assistantMessage(usage({ cacheWrite: 100_000 })),
			{ role: "branchSummary", summary: "older branch", fromId: "old", timestamp: 2 },
			userMessage("newer work"),
			assistantMessage(usage({ input: 2_000 }), { timestamp: 3 }),
		];
		// Warm: the probe reads back the 2k parent baseline.
		expect(
			detectBranchSummaryCacheMiss(history, summaryMessage(usage({ input: 2_000, cacheRead: 61_000 })), models),
		).toBeUndefined();
	});

	it("still skips cache-less providers across a branch summary", () => {
		const history: AgentMessage[] = [
			assistantMessage(usage({ input: 100_000 })),
			{ role: "branchSummary", summary: "older branch", fromId: "old", timestamp: 2 },
			userMessage("newer work"),
			assistantMessage(usage({ input: 2_000 }), { timestamp: 3 }),
		];
		expect(detectBranchSummaryCacheMiss(history, summaryMessage(usage({ input: 60_000 })), models)).toBeUndefined();
	});

	it("keeps session cache capability across a reset (E2E: earlier summary hit, later full miss)", () => {
		const history: AgentMessage[] = [
			assistantMessage(usage({ cacheWrite: 100_000 })),
			{ role: "compactionSummary", summary: "compacted", tokensBefore: 1_000, timestamp: 2 },
			userMessage("newer work"),
			assistantMessage(usage({ input: 2_000 }), { timestamp: 3 }),
		];
		// Cache capability is session-scoped: the pre-reset write proves the
		// provider reports caching, so the within-segment zero-read still counts.
		const miss = detectBranchSummaryCacheMiss(history, summaryMessage(usage({ input: 60_000 })), models);
		expect(miss?.missedTokens).toBe(2_000);
	});

	it("still skips cache-less providers across a reset", () => {
		const history: AgentMessage[] = [
			assistantMessage(usage({ input: 100_000 })),
			{ role: "compactionSummary", summary: "compacted", tokensBefore: 1_000, timestamp: 2 },
			userMessage("newer work"),
			assistantMessage(usage({ input: 2_000 }), { timestamp: 3 }),
		];
		expect(detectBranchSummaryCacheMiss(history, summaryMessage(usage({ input: 60_000 })), models)).toBeUndefined();
	});

	it("stays silent on a cold response after a model switch (expected re-billing)", () => {
		const miss = detectBranchSummaryCacheMiss(
			warmedHistory,
			{ ...summaryMessage(usage({ input: 100_000 })), model: "other-model" },
			models,
		);
		// A new model cannot read the previous prefix: expected re-billing,
		// not an actionable miss.
		expect(miss).toBeUndefined();
	});

	it.each([
		{ cacheRead: 100_000 - 1_024, expected: undefined },
		{ cacheRead: 100_000 - 1_025, expected: 1_025 },
	])("noise boundary: cacheRead $cacheRead", ({ cacheRead, expected }) => {
		const miss = detectBranchSummaryCacheMiss(
			warmedHistory,
			summaryMessage(usage({ input: 100_000, cacheRead })),
			models,
		);
		if (expected === undefined) expect(miss).toBeUndefined();
		else expect(miss?.missedTokens).toBe(expected);
	});

	it("stays silent on a warm response after a model switch", () => {
		// Silent via provider-switch suppression (as well as the warm read-back).
		expect(
			detectBranchSummaryCacheMiss(
				warmedHistory,
				{ ...summaryMessage(usage({ input: 2_000, cacheRead: 100_000 })), model: "other-model" },
				models,
			),
		).toBeUndefined();
	});
});

describe("shouldDisplayBranchSummaryCacheMiss", () => {
	it.each([
		{ missedTokens: 19_999, missedCost: 0.09, expected: false },
		{ missedTokens: 20_000, missedCost: 0, expected: true },
		{ missedTokens: 19_999, missedCost: 0.1, expected: true },
		// Cost-only trigger: small miss, large overcharge.
		{ missedTokens: 5_000, missedCost: 0.15, expected: true },
		// Tokens-only trigger: large miss, unknown pricing.
		{ missedTokens: 25_000, missedCost: 0, expected: true },
	])("display boundary: $missedTokens tokens + $$$missedCost", ({ missedTokens, missedCost, expected }) => {
		expect(shouldDisplayBranchSummaryCacheMiss({ missedTokens, missedCost, idleMs: 0, modelChanged: false })).toBe(
			expected,
		);
	});
});
