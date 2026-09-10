import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { AgentMessage } from "../../types.ts";

/** Per-turn misses at or below this are cache breakpoint granularity noise. */
const NOISE_FLOOR_TOKENS = 1024;

/** Display threshold: misses below this many tokens stay silent unless costly. */
export const CACHE_MISS_DISPLAY_TOKENS = 20_000;
/** Display threshold: misses below this extra cost stay silent unless large. */
export const CACHE_MISS_DISPLAY_COST = 0.1;

/**
 * Whether a measured miss is worth displaying. Mirrors the classic
 * addCacheMissNotice thresholds: below both limits the notice stays silent.
 * Renderers additionally gate on their own showCacheMissNotices setting.
 */
export function shouldDisplayBranchSummaryCacheMiss(miss: BranchSummaryCacheMiss): boolean {
	return miss.missedTokens >= CACHE_MISS_DISPLAY_TOKENS || miss.missedCost >= CACHE_MISS_DISPLAY_COST;
}

/**
 * Measured prompt-cache miss paid by a branch-summary request.
 * Same JSON shape as the coding-agent BranchSummaryCacheMiss; persisted on
 * the branch_summary entry so transcript renderers can re-render the notice.
 * Display metadata only: never fed back into billed-token accounting.
 */
export interface BranchSummaryCacheMiss {
	/** Prompt tokens that were in the previous turn's prompt but not read from cache. */
	missedTokens: number;
	/** Extra dollars paid vs. a full cache hit; 0 when pricing is unknown. */
	missedCost: number;
	/** Milliseconds since the previous request (which last refreshed the cache). */
	idleMs: number;
	/** True when the model changed relative to the previous request. */
	modelChanged: boolean;
}

/** Minimal pricing lookup; satisfied by Models and ModelRuntime alike. Cost is $/million tokens. */
export interface ModelPriceSource {
	getModel(provider: string, modelId: string): { cost: { cacheRead: number } } | undefined;
}

/** The last request seen by the scan; everything in its prompt should be cached. */
interface PreviousRequest {
	promptTokens: number;
	modelKey: string;
	timestamp: number;
	/**
	 * Sticky: some earlier request in this scan segment reported cache activity.
	 * Distinguishes a total miss on a cache-read-only provider (OpenAI-style,
	 * writes unreported) from a provider that never reports caching at all.
	 */
	reportedCache: boolean;
}

/**
 * Compute the cache miss for the branch-summary response relative to the
 * previous request. Returns undefined when nothing is counted: first turn,
 * after a reset, no cache activity ever reported (provider without cache
 * support), or miss below the noise floor. Same thresholds as the classic
 * detectCacheMiss scan; the legacy retention-none path is not special-cased
 * and always misses here on measurement.
 */
function detectMiss(
	prev: PreviousRequest | undefined,
	response: AssistantMessage,
	models: ModelPriceSource,
): BranchSummaryCacheMiss | undefined {
	const usage = response.usage;
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
	// from this response's own cost breakdown.
	const paidTokens = usage.input + usage.cacheWrite;
	const paidPerToken = paidTokens > 0 ? (usage.cost.input + usage.cost.cacheWrite) / paidTokens : 0;
	const readPerToken =
		usage.cacheRead > 0
			? usage.cost.cacheRead / usage.cacheRead
			: (models.getModel(response.provider, response.model)?.cost.cacheRead ?? 0) / 1_000_000;

	return {
		missedTokens,
		missedCost: missedTokens * Math.max(0, paidPerToken - readPerToken),
		idleMs: Math.max(0, response.timestamp - prev.timestamp),
		modelChanged: `${response.provider}/${response.model}` !== prev.modelKey,
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

/**
 * Detect a cache miss on a just-completed branch-summary response.
 * `messages` is the chronological preparation history (background + branch);
 * prior summary messages reset the scan the same way compaction and
 * branch_summary entries reset the classic scan.
 */
export function detectBranchSummaryCacheMiss(
	messages: readonly AgentMessage[],
	response: AssistantMessage,
	models: ModelPriceSource,
): BranchSummaryCacheMiss | undefined {
	let prev: PreviousRequest | undefined;
	for (const message of messages) {
		if (message.role === "branchSummary" || message.role === "compactionSummary") {
			// The context legitimately changed; the summary prompt after this
			// is new content, not re-billed content.
			prev = undefined;
			continue;
		}
		if (message.role === "assistant") {
			prev = asPreviousRequest(message, prev?.reportedCache ?? false) ?? prev;
		}
	}
	return detectMiss(prev, response, models);
}
