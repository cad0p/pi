import type { Usage } from "@earendil-works/pi-ai";
import { Container } from "@earendil-works/pi-tui";
import { describe, expect, test, vi } from "vitest";
import type { CacheMiss } from "../src/core/cache-stats.ts";
import type { SessionEntry } from "../src/core/session-manager.ts";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";
import { stripAnsi } from "../src/utils/ansi.ts";

describe("InteractiveMode compaction events", () => {
	test("uses the cache miss notice setting for compaction and branch summary costs", () => {
		const usage: Usage = {
			input: 10,
			output: 20,
			cacheRead: 30,
			cacheWrite: 40,
			totalTokens: 100,
			cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.065, total: 0.125 },
		};
		const addCompactionCostNotice = Reflect.get(InteractiveMode.prototype, "addCompactionCostNotice") as (
			this: { chatContainer: Container; settingsManager: { getShowCacheMissNotices(): boolean } },
			notice: {
				type: "compaction_cost";
				kind: "compaction" | "branch_summary";
				usage: Usage;
			},
		) => void;

		initTheme("dark");
		const enabled = {
			chatContainer: new Container(),
			settingsManager: { getShowCacheMissNotices: () => true },
		};
		addCompactionCostNotice.call(enabled, { type: "compaction_cost", kind: "compaction", usage });
		addCompactionCostNotice.call(enabled, {
			type: "compaction_cost",
			kind: "branch_summary",
			usage,
		});
		const output = stripAnsi(enabled.chatContainer.render(120).join("\n"));
		expect(output).toContain("Compaction: 100 tokens billed (70 cache hit) (~$0.13)");

		const disabled = {
			chatContainer: new Container(),
			settingsManager: { getShowCacheMissNotices: () => false },
		};
		addCompactionCostNotice.call(disabled, { type: "compaction_cost", kind: "compaction", usage });
		expect(disabled.chatContainer.children).toHaveLength(0);
	});

	test("renders each compaction cost after its summary", () => {
		const currentUsage: Usage = {
			input: 10,
			output: 20,
			cacheRead: 30,
			cacheWrite: 40,
			totalTokens: 100,
			cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.04, total: 0.1 },
		};
		const previousUsage: Usage = {
			input: 1,
			output: 2,
			cacheRead: 3,
			cacheWrite: 4,
			totalTokens: 10,
			cost: { input: 0.001, output: 0.002, cacheRead: 0.003, cacheWrite: 0.004, total: 0.01 },
		};
		const entries: SessionEntry[] = [
			{
				type: "compaction",
				id: "current",
				parentId: "previous",
				timestamp: "2025-01-02T00:00:00Z",
				summary: "current summary",
				firstKeptEntryId: "kept",
				tokensBefore: 200,
				usage: currentUsage,
			},
			{
				type: "compaction",
				id: "previous",
				parentId: null,
				timestamp: "2025-01-01T00:00:00Z",
				summary: "previous summary",
				firstKeptEntryId: "kept",
				tokensBefore: 100,
				usage: previousUsage,
			},
		];
		const fakeThis = { renderSessionItems: vi.fn() };
		const renderSessionEntries = Reflect.get(InteractiveMode.prototype, "renderSessionEntries") as (
			this: typeof fakeThis,
			entries: SessionEntry[],
		) => void;

		renderSessionEntries.call(fakeThis, entries);

		expect(fakeThis.renderSessionItems).toHaveBeenCalledWith(
			[
				expect.objectContaining({ role: "compactionSummary", summary: "current summary" }),
				{ type: "compaction_cost", kind: "compaction", usage: currentUsage },
				expect.objectContaining({ role: "compactionSummary", summary: "previous summary" }),
				{ type: "compaction_cost", kind: "compaction", usage: previousUsage },
			],
			{},
		);
	});

	test("renders retained entries and appends the latest summary cost at the bottom", async () => {
		const usage: Usage = {
			input: 10,
			output: 20,
			cacheRead: 30,
			cacheWrite: 40,
			totalTokens: 100,
			cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.065, total: 0.125 },
		};
		const latestCompaction: SessionEntry = {
			type: "compaction",
			id: "latest",
			parentId: "previous",
			timestamp: "2025-01-02T00:00:00Z",
			summary: "summary",
			firstKeptEntryId: "kept",
			tokensBefore: 123,
			usage,
		};
		const previousCompaction: SessionEntry = {
			type: "compaction",
			id: "previous",
			parentId: null,
			timestamp: "2025-01-01T00:00:00Z",
			summary: "previous summary",
			firstKeptEntryId: "kept",
			tokensBefore: 100,
			usage,
		};
		const fakeThis = {
			isInitialized: true,
			footer: { invalidate: vi.fn() },
			autoCompactionEscapeHandler: undefined as (() => void) | undefined,
			autoCompactionLoader: undefined,
			defaultEditor: {},
			statusContainer: { clear: vi.fn() },
			chatContainer: { clear: vi.fn() },
			sessionManager: { buildContextEntries: vi.fn().mockReturnValue([latestCompaction, previousCompaction]) },
			renderSessionEntries: vi.fn(),
			addMessageToChat: vi.fn(),
			addCompactionCostNotice: vi.fn(),
			showError: vi.fn(),
			showStatus: vi.fn(),
			clearStatusIndicator: vi.fn(),
			flushCompactionQueue: vi.fn().mockResolvedValue(undefined),
			settingsManager: { getShowTerminalProgress: () => false },
			ui: { requestRender: vi.fn(), terminal: { setProgress: vi.fn() } },
		};

		const handleEvent = Reflect.get(InteractiveMode.prototype, "handleEvent") as (
			this: typeof fakeThis,
			event: {
				type: "compaction_end";
				reason: "manual" | "threshold" | "overflow";
				result: { tokensBefore: number; summary: string; usage?: Usage } | undefined;
				aborted: boolean;
				willRetry: boolean;
				errorMessage?: string;
			},
		) => Promise<void>;

		await handleEvent.call(fakeThis, {
			type: "compaction_end",
			reason: "manual",
			result: {
				tokensBefore: 123,
				summary: "summary",
				usage,
			},
			aborted: false,
			willRetry: false,
		});

		expect(fakeThis.chatContainer.clear).toHaveBeenCalledTimes(1);
		expect(fakeThis.renderSessionEntries).toHaveBeenCalledWith([previousCompaction]);
		expect(fakeThis.addMessageToChat).toHaveBeenCalledTimes(1);
		expect(fakeThis.addMessageToChat).toHaveBeenCalledWith(
			expect.objectContaining({
				role: "compactionSummary",
				tokensBefore: 123,
				summary: "summary",
			}),
		);
		expect(fakeThis.addCompactionCostNotice).toHaveBeenCalledWith({
			type: "compaction_cost",
			kind: "compaction",
			usage,
		});
		expect(fakeThis.flushCompactionQueue).toHaveBeenCalledWith({ willRetry: false });
	});

	test("carries persisted branch-summary misses into rebuild notices", () => {
		const usage: Usage = {
			input: 10,
			output: 20,
			cacheRead: 30,
			cacheWrite: 40,
			totalTokens: 100,
			cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.04, total: 0.1 },
		};
		const cacheMiss: CacheMiss = { missedTokens: 105_000, missedCost: 0.36, idleMs: 0, modelChanged: false };
		const entries: SessionEntry[] = [
			{
				type: "branch_summary",
				id: "with-miss",
				parentId: null,
				timestamp: "2025-01-02T00:00:00Z",
				fromId: "old-leaf",
				summary: "summary with a measured miss",
				usage,
				cacheMiss,
			},
			{
				type: "branch_summary",
				id: "without-miss",
				parentId: "with-miss",
				timestamp: "2025-01-03T00:00:00Z",
				fromId: "other-leaf",
				summary: "warm summary",
				usage,
			},
		];
		const fakeThis = { renderSessionItems: vi.fn() };
		const renderSessionEntries = Reflect.get(InteractiveMode.prototype, "renderSessionEntries") as (
			this: typeof fakeThis,
			entries: SessionEntry[],
		) => void;

		renderSessionEntries.call(fakeThis, entries);

		expect(fakeThis.renderSessionItems).toHaveBeenCalledWith(
			[
				expect.objectContaining({ role: "branchSummary", summary: "summary with a measured miss" }),
				{ type: "compaction_cost", kind: "branch_summary", usage, cacheMiss },
				expect.objectContaining({ role: "branchSummary", summary: "warm summary" }),
				{ type: "compaction_cost", kind: "branch_summary", usage },
			],
			{},
		);
	});

	test("re-renders persisted branch-summary misses on rebuild when enabled, silent when off", () => {
		const usage: Usage = {
			input: 10,
			output: 20,
			cacheRead: 30,
			cacheWrite: 40,
			totalTokens: 100,
			cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.04, total: 0.1 },
		};
		const cacheMiss: CacheMiss = { missedTokens: 105_000, missedCost: 0.36, idleMs: 0, modelChanged: false };
		const notice = { type: "compaction_cost", kind: "branch_summary", usage, cacheMiss };
		const renderSessionItems = Reflect.get(InteractiveMode.prototype, "renderSessionItems") as (
			this: {
				pendingTools: Map<string, unknown>;
				settingsManager: { getShowCacheMissNotices(): boolean };
				session: { modelRuntime: { getModel(): undefined } };
				sessionManager: { getEntries(): SessionEntry[] };
				chatContainer: unknown;
				ui: { requestRender(): void };
				addCompactionCostNotice(notice: unknown): void;
				addCacheMissNotice(miss: CacheMiss): void;
			},
			items: unknown[],
		) => void;
		const fakeThis = (enabled: boolean) => ({
			pendingTools: new Map<string, unknown>(),
			settingsManager: { getShowCacheMissNotices: () => enabled },
			session: { modelRuntime: { getModel: () => undefined } },
			sessionManager: { getEntries: () => [] as SessionEntry[] },
			chatContainer: {},
			ui: { requestRender: vi.fn() },
			addCompactionCostNotice: vi.fn(),
			addCacheMissNotice: vi.fn(),
		});

		// Rebuild path (renderInitialMessages after chat.clear funnels here):
		// branch summaries follow the package pattern (hits read out in the
		// footer) — no billed notice, only the persisted miss re-renders.
		const enabled = fakeThis(true);
		renderSessionItems.call(enabled, [notice]);
		expect(enabled.addCompactionCostNotice).not.toHaveBeenCalled();
		expect(enabled.addCacheMissNotice).toHaveBeenCalledWith(cacheMiss);

		// Setting off: neither notice renders.
		const disabled = fakeThis(false);
		renderSessionItems.call(disabled, [notice]);
		expect(disabled.addCompactionCostNotice).not.toHaveBeenCalled();
		expect(disabled.addCacheMissNotice).not.toHaveBeenCalled();
	});

	test("renders persisted branch-summary misses with the live miss copy and thresholds", () => {
		const addCacheMissNotice = Reflect.get(InteractiveMode.prototype, "addCacheMissNotice") as (
			this: { chatContainer: Container },
			miss: CacheMiss,
		) => void;

		initTheme("dark");
		const shown = { chatContainer: new Container() };
		addCacheMissNotice.call(shown, {
			missedTokens: 105_000,
			missedCost: 0.36,
			idleMs: 0,
			modelChanged: false,
		});
		const output = stripAnsi(shown.chatContainer.render(120).join("\n"));
		expect(output).toContain("Cache miss");
		expect(output).toContain("re-billed");

		const switched = { chatContainer: new Container() };
		addCacheMissNotice.call(switched, {
			missedTokens: 105_000,
			missedCost: 0.36,
			idleMs: 0,
			modelChanged: true,
		});
		expect(stripAnsi(switched.chatContainer.render(120).join("\n"))).toContain("Cache miss after model switch");

		// Below the display thresholds the persisted miss stays silent.
		const quiet = { chatContainer: new Container() };
		addCacheMissNotice.call(quiet, { missedTokens: 500, missedCost: 0, idleMs: 0, modelChanged: false });
		expect(quiet.chatContainer.children).toHaveLength(0);
	});

	test.each([
		{ missedTokens: 19_999, missedCost: 0.09, shown: false },
		{ missedTokens: 20_000, missedCost: 0, shown: true },
		{ missedTokens: 19_999, missedCost: 0.1, shown: true },
		// Cost-only trigger: small miss, large overcharge.
		{ missedTokens: 5_000, missedCost: 0.15, shown: true },
		// Tokens-only trigger: large miss, unknown pricing.
		{ missedTokens: 25_000, missedCost: 0, shown: true },
	])("gates persisted misses at $missedTokens tokens + $$$missedCost", ({ missedTokens, missedCost, shown }) => {
		const addCacheMissNotice = Reflect.get(InteractiveMode.prototype, "addCacheMissNotice") as (
			this: { chatContainer: Container },
			miss: CacheMiss,
		) => void;
		initTheme("dark");
		const fakeThis = { chatContainer: new Container() };
		addCacheMissNotice.call(fakeThis, { missedTokens, missedCost, idleMs: 0, modelChanged: false });
		if (shown) {
			expect(stripAnsi(fakeThis.chatContainer.render(120).join("\n"))).toContain("Cache miss");
		} else {
			expect(fakeThis.chatContainer.children).toHaveLength(0);
		}
	});

	test("updates the working state when the same agent run resumes after compaction", async () => {
		const fakeThis = {
			isInitialized: true,
			footer: { invalidate: vi.fn() },
			activeStatusIndicator: undefined,
			workingVisible: true,
			showWorkingStatusIndicator: vi.fn(),
			clearStatusIndicator: vi.fn(),
			settingsManager: { getShowTerminalProgress: () => true },
			ui: { requestRender: vi.fn(), terminal: { setProgress: vi.fn() } },
		};
		const handleEvent = Reflect.get(InteractiveMode.prototype, "handleEvent") as (
			this: typeof fakeThis,
			event: { type: "turn_start" },
		) => Promise<void>;

		await handleEvent.call(fakeThis, { type: "turn_start" });

		expect(fakeThis.ui.terminal.setProgress).toHaveBeenCalledWith(true);
		expect(fakeThis.showWorkingStatusIndicator).toHaveBeenCalledTimes(1);
		expect(fakeThis.clearStatusIndicator).not.toHaveBeenCalled();
		expect(fakeThis.ui.requestRender).toHaveBeenCalledTimes(1);

		fakeThis.workingVisible = false;
		await handleEvent.call(fakeThis, { type: "turn_start" });

		expect(fakeThis.showWorkingStatusIndicator).toHaveBeenCalledTimes(1);
		expect(fakeThis.clearStatusIndicator).toHaveBeenCalledTimes(1);
		expect(fakeThis.ui.requestRender).toHaveBeenCalledTimes(2);
	});

	test("preserves steering behavior when flushing into an active agent run", async () => {
		const fakeThis = {
			compactionQueuedMessages: [{ text: "change direction", mode: "steer" as const }],
			session: {
				clearQueue: vi.fn(),
				prompt: vi.fn().mockResolvedValue(undefined),
				steer: vi.fn().mockResolvedValue(undefined),
				followUp: vi.fn().mockResolvedValue(undefined),
			},
			isExtensionCommand: vi.fn().mockReturnValue(false),
			updatePendingMessagesDisplay: vi.fn(),
			showError: vi.fn(),
		};

		const flushCompactionQueue = Reflect.get(InteractiveMode.prototype, "flushCompactionQueue") as (
			this: typeof fakeThis,
			options?: { willRetry?: boolean },
		) => Promise<void>;

		await flushCompactionQueue.call(fakeThis, { willRetry: false });

		expect(fakeThis.session.prompt).toHaveBeenCalledWith("change direction", { streamingBehavior: "steer" });
		expect(fakeThis.compactionQueuedMessages).toEqual([]);
		expect(fakeThis.showError).not.toHaveBeenCalled();
	});
});
