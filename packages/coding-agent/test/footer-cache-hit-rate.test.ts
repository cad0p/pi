import type { Usage } from "@earendil-works/pi-ai";
import { describe, expect, test } from "vitest";
import type { AgentSession } from "../src/core/agent-session.ts";
import type { SessionEntry } from "../src/core/session-manager.ts";
import { FooterComponent } from "../src/modes/interactive/components/footer.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";
import { stripAnsi } from "../src/utils/ansi.ts";

function usage(input: number, cacheRead: number): Usage {
	return {
		input,
		output: 10,
		cacheRead,
		cacheWrite: 0,
		totalTokens: input + 10 + cacheRead,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function assistantEntry(id: string, timestamp: string, input: number, cacheRead: number): SessionEntry {
	return {
		type: "message",
		id,
		parentId: null,
		timestamp,
		message: {
			role: "assistant",
			content: [{ type: "text", text: "hi" }],
			api: "openai-responses",
			provider: "opencode-go",
			model: "muse-spark",
			usage: usage(input, cacheRead),
			stopReason: "stop",
			timestamp: Date.parse(timestamp),
		},
	} as unknown as SessionEntry;
}

function summaryEntry(id: string, timestamp: string, input: number, cacheRead: number): SessionEntry {
	return {
		type: "branch_summary",
		id,
		parentId: null,
		timestamp,
		fromId: "branch-tip",
		summary: "summary",
		usage: usage(input, cacheRead),
	} as unknown as SessionEntry;
}

function renderFooter(entries: SessionEntry[]): string {
	initTheme("dark");
	const session = {
		state: {},
		sessionManager: {
			getEntries: () => entries,
			getCwd: () => "/tmp",
			getSessionName: () => undefined,
		},
		getContextUsage: () => undefined,
		modelRuntime: { isUsingSubscription: () => false },
	} as unknown as AgentSession;
	const footerData = {
		getGitBranch: () => null,
		getExtensionStatuses: () => new Map(),
		getAvailableProviderCount: () => 1,
		onBranchChange: () => () => {},
	};
	const footer = new FooterComponent(session, footerData);
	return stripAnsi(footer.render(200).join("\n"));
}

describe("footer CH rate", () => {
	test("reflects a newer branch summary instead of an older assistant turn", () => {
		// E2E figures: older turn at 99.3%, newer warm summary at 61k/63k = 96.8%.
		const output = renderFooter([
			assistantEntry("a", "2026-09-10T02:00:00.000Z", 700, 99300),
			summaryEntry("s", "2026-09-10T03:00:00.000Z", 2000, 61000),
		]);
		expect(output).toContain("CH96.8%");
		expect(output).not.toContain("CH99.3%");
	});

	test("keeps the assistant turn rate when the summary is older", () => {
		const output = renderFooter([
			summaryEntry("s", "2026-09-10T02:00:00.000Z", 2000, 61000),
			assistantEntry("a", "2026-09-10T03:00:00.000Z", 700, 99300),
		]);
		expect(output).toContain("CH99.3%");
		expect(output).not.toContain("CH96.8%");
	});
});
