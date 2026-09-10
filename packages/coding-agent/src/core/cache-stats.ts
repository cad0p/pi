import type { AssistantMessage, Usage } from "@earendil-works/pi-ai";
import type { SessionEntry } from "./session-manager.ts";

/**
 * Prompt-cache TTL: idle gaps longer than this are worth mentioning as the
 * likely cause of a miss. Anthropic's default cache TTL is 5 minutes.
 */
export const CACHE_TTL_MS = 5 * 60 * 1000;

/** Per-turn misses at or below this are cache breakpoint granularity noise. */
const NOISE_FLOOR_TOKENS = 1024;

/** A counted cache miss on a single assistant message. */
export interface CacheMiss {
	/** Prompt tokens that were in the previous turn's prompt but not read from cache. */
	missedTokens: number;
	/** Extra dollars paid vs. a full cache hit; 0 when pricing is unknown. */
	missedCost: number;
	/** Milliseconds since the previous request (which last refreshed the cache). */
	idleMs: number;
	/** True when the model changed relative to the previous request. */
	modelChanged: boolean;
}

export interface CacheWasteTotals {
	missedTokens: number;
	missedCost: number;
	/** Number of counted misses (turns above the noise floor). */
	missCount: number;
}

/** Minimal pricing lookup, satisfied by ModelRuntime. Cost is $/million tokens. */
export interface ModelPriceSource {
	getModel(provider: string, modelId: string): { cost: { cacheRead: number } } | undefined;
}

/** The last request seen by the scan; everything in its prompt should be cached. */
interface PreviousRequest {
	promptTokens: number;
	modelKey: string;
	timestamp: number;
	/**
	 * Sticky: some earlier request in this session reported cache activity.
	 * Session-scoped (never reset by context boundaries): provider cache
	 * capability does not change across compactions, while the prompt
	 * baseline legitimately does. Distinguishes a total miss on a
	 * cache-read-only provider (OpenAI-style, writes unreported) from a
	 * provider that never reports caching at all.
	 */
	reportedCache: boolean;
}

/**
 * Compute the cache miss for one assistant message relative to the previous
 * request. Returns undefined when nothing is counted: first turn, after a
 * reset, no cache activity ever reported (provider without cache support), or
 * miss below the noise floor.
 */
function detectMiss(
	prev: PreviousRequest | undefined,
	message: Pick<AssistantMessage, "provider" | "model" | "usage" | "timestamp">,
	models: ModelPriceSource,
): CacheMiss | undefined {
	const usage = message.usage;
	const promptTokens = usage.input + usage.cacheRead + usage.cacheWrite;
	// A zero-cache turn only counts when cache activity was reported before:
	// on cache-read-only providers that is a total miss, while on providers
	// that never report caching it means nothing.
	if (!prev || promptTokens <= 0 || (usage.cacheRead + usage.cacheWrite === 0 && !prev.reportedCache)) {
		return undefined;
	}

	const missedTokens = Math.min(prev.promptTokens, promptTokens) - usage.cacheRead;
	if (missedTokens <= NOISE_FLOOR_TOKENS) return undefined;

	// Extra cost = missed tokens billed at the actual paid rate (input/cacheWrite,
	// incl. write premium) instead of the cache-read rate. Missed tokens can only
	// land in the input or cacheWrite buckets, so the paid rate comes straight
	// from this message's own cost breakdown.
	const paidTokens = usage.input + usage.cacheWrite;
	const paidPerToken = paidTokens > 0 ? (usage.cost.input + usage.cost.cacheWrite) / paidTokens : 0;
	const readPerToken =
		usage.cacheRead > 0
			? usage.cost.cacheRead / usage.cacheRead
			: (models.getModel(message.provider, message.model)?.cost.cacheRead ?? 0) / 1_000_000;

	return {
		missedTokens,
		missedCost: missedTokens * Math.max(0, paidPerToken - readPerToken),
		idleMs: Math.max(0, message.timestamp - prev.timestamp),
		modelChanged: `${message.provider}/${message.model}` !== prev.modelKey,
	};
}

function asPreviousRequest(message: AssistantMessage, reportedCache: boolean): PreviousRequest | undefined {
	const usage = message.usage;
	const promptTokens = usage.input + usage.cacheRead + usage.cacheWrite;
	if (promptTokens <= 0) return undefined;
	return {
		promptTokens,
		modelKey: `${message.provider}/${message.model}`,
		timestamp: message.timestamp,
		reportedCache: reportedCache || usage.cacheRead + usage.cacheWrite > 0,
	};
}

function scan(
	entries: SessionEntry[],
	models: ModelPriceSource,
	// Probe-only override: summary probes pass true so the parent baseline
	// survives branch summaries (their request reuses the live prefix — see
	// generateBranchSummaryWithRequest in compaction/branch-summarization.ts).
	// Live-turn accounting always uses the default false.
	keepBaselineAcrossBranchSummary = false,
): { prev: PreviousRequest | undefined; totals: CacheWasteTotals; misses: Map<AssistantMessage, CacheMiss> } {
	let prev: PreviousRequest | undefined;
	const totals: CacheWasteTotals = { missedTokens: 0, missedCost: 0, missCount: 0 };
	const misses = new Map<AssistantMessage, CacheMiss>();

	// Session-level cache capability: any measured cache activity (assistant
	// turns AND summary requests) proves the provider reports caching, so a
	// later zero-read is a real miss even across a context boundary.
	let everReportedCache = false;
	for (const entry of entries) {
		if (entry.type === "compaction" || (entry.type === "branch_summary" && !keepBaselineAcrossBranchSummary)) {
			// The context legitimately changed; the next turn's prompt is new content,
			// not re-billed content. Model switches are NOT exempt: they re-bill the
			// full prompt and should be counted.
			if (entry.usage && entry.usage.cacheRead + entry.usage.cacheWrite > 0) {
				everReportedCache = true;
			}
			prev = undefined;
			continue;
		}
		if (entry.type === "branch_summary") {
			// Probe-only path (keepBaselineAcrossBranchSummary): the summary
			// request reuses the live prompt-cache prefix, so the parent baseline
			// survives. Fold cache activity into the session capability flag but
			// never reset prev and never become prev (only assistant messages do).
			if (entry.usage && entry.usage.cacheRead + entry.usage.cacheWrite > 0) {
				everReportedCache = true;
			}
			continue;
		}
		if (entry.type === "message" && entry.message.role === "assistant") {
			const miss = detectMiss(prev, entry.message, models);
			if (miss) {
				totals.missedTokens += miss.missedTokens;
				totals.missedCost += miss.missedCost;
				totals.missCount += 1;
				misses.set(entry.message, miss);
			}
			if (entry.message.usage.cacheRead + entry.message.usage.cacheWrite > 0) {
				everReportedCache = true;
			}
			prev = asPreviousRequest(entry.message, (prev?.reportedCache ?? false) || everReportedCache) ?? prev;
		}
	}
	return { prev, totals, misses };
}

/**
 * Cumulative cache waste across a session: prompt tokens that should have been
 * cache reads (they were in the previous turn's prompt) but were re-billed.
 */
export function computeCacheWaste(entries: SessionEntry[], models: ModelPriceSource): CacheWasteTotals {
	return scan(entries, models).totals;
}

/**
 * All counted cache misses across a session, keyed by the assistant message
 * (by reference) that paid for them. Used to re-derive transcript notices when
 * rebuilding the chat from entries (resume, post-compaction rebuild).
 */
export function collectCacheMisses(
	entries: SessionEntry[],
	models: ModelPriceSource,
): Map<AssistantMessage, CacheMiss> {
	return scan(entries, models).misses;
}

/**
 * Detect a cache miss on a just-completed assistant message.
 * `entries` must not yet contain `message` (message_end fires before persistence).
 */
export function detectCacheMiss(
	entries: SessionEntry[],
	message: AssistantMessage,
	models: ModelPriceSource,
): CacheMiss | undefined {
	return detectMiss(scan(entries, models).prev, message, models);
}

/**
 * Detect a cache miss on a just-completed branch-summary response from its
 * measured usage. `entries` is the session before the summary entry is
 * appended. Production seam for the branch-summary generation sites (classic
 * AgentSession.navigateTree); the legacy retention-none path flows through
 * the same measurement with no per-path special-casing.
 */
export function detectBranchSummaryCacheMiss(
	entries: SessionEntry[],
	responseUsage: Usage,
	provider: string,
	model: string,
	timestamp: number,
	models: ModelPriceSource,
): CacheMiss | undefined {
	const { prev } = scan(entries, models, true);
	// Live-turn accounting counts model switches as misses; summary probes
	// suppress them instead. A cold summary right after a switch is expected
	// re-billing (the new provider/model cannot read the previous prefix),
	// not an actionable miss — warning would spam on config-driven switches.
	if (prev && prev.modelKey !== `${provider}/${model}`) return undefined;
	return detectMiss(prev, { provider, model, usage: responseUsage, timestamp }, models);
}
