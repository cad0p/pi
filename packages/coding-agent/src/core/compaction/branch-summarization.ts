/**
 * Branch summarization for tree navigation.
 *
 * When navigating to a different point in the session tree, this generates
 * a summary of the branch being left so context isn't lost.
 */

import type { AgentMessage, AgentTool, StreamFn, ThinkingLevel } from "@earendil-works/pi-agent-core";
import type { RetryCallbacks, RetryPolicy } from "@earendil-works/pi-ai";
import { contentText, type Message } from "@earendil-works/pi-ai";
import type { Model, SimpleStreamOptions, Usage } from "@earendil-works/pi-ai/compat";
import {
	convertToLlm,
	createBranchSummaryMessage,
	createCompactionSummaryMessage,
	createCustomMessage,
} from "../messages.ts";
import { type ReadonlySessionManager, type SessionEntry, sessionEntryToContextMessages } from "../session-manager.ts";
import { completeSummarization, estimateTokens, getSummarizationFailure } from "./compaction.ts";
import {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	type FileOperations,
	formatFileOperations,
	SUMMARIZATION_SYSTEM_PROMPT,
	serializeConversation,
	stripBoundaryOrphanToolResults,
} from "./utils.ts";

// ============================================================================
// Types
// ============================================================================

export interface BranchSummaryResult {
	summary?: string;
	usage?: Usage;
	readFiles?: string[];
	modifiedFiles?: string[];
	aborted?: boolean;
	error?: string;
}

/** Details stored in BranchSummaryEntry.details for file tracking */
export interface BranchSummaryDetails {
	readFiles: string[];
	modifiedFiles: string[];
}

export type { FileOperations } from "./utils.ts";

export interface BranchPreparation {
	/** Messages extracted for summarization, in chronological order */
	messages: AgentMessage[];
	/** File operations extracted from tool calls */
	fileOps: FileOperations;
	/** Total estimated tokens in messages */
	totalTokens: number;
	/**
	 * 1-based number of the first branch message within `messages`
	 * (message numbering starts at 1 and excludes the system prompt).
	 * Pre-branch background messages precede it; they are sent so the
	 * request shares the live turns' cache prefix, not to be summarized.
	 */
	firstMessageNumber: number;
}

export interface CollectEntriesResult {
	/** Entries to summarize, in chronological order */
	entries: SessionEntry[];
	/** Common ancestor between old and new position, if any */
	commonAncestorId: string | null;
	/**
	 * Pre-branch background entries (root-first, ending at the common
	 * ancestor), in chronological order. Sent with the cache-preserving
	 * request so its prefix matches live turns; never summarized.
	 */
	prefixEntries: SessionEntry[];
}

export interface GenerateBranchSummaryOptions {
	/** Model to use for summarization */
	model: Model<any>;
	/** API key for the model */
	apiKey?: string;
	/** Request headers for the model */
	headers?: Record<string, string>;
	/** Provider-scoped environment values for the model */
	env?: Record<string, string>;
	/** Abort signal for cancellation */
	signal: AbortSignal;
	/** Optional custom instructions for summarization */
	customInstructions?: string;
	/** If true, customInstructions replaces the default prompt instead of being appended */
	replaceInstructions?: boolean;
	/** Tokens reserved when selecting branch history (default 16384) */
	reserveTokens?: number;
	/** Optional session stream function. Used to preserve SDK request behavior without mutating agent state. */
	streamFn?: StreamFn;
	/** Retry policy for transient summarization errors. Reuses coding-agent's `settings.retry`. */
	retry?: RetryPolicy;
	/** Optional callbacks for retry reporting (e.g. TUI retry indicators). */
	callbacks?: RetryCallbacks;
	/**
	 * Live thinking level, forwarded as the request reasoning effort on the
	 * cache-preserving path only. Live turns send their thinking level
	 * explicitly and the provider keys cache on it — a summary that omits
	 * it runs at a different effort and can never hit the live prefix.
	 * The legacy path never sent reasoning and is unchanged.
	 */
	thinkingLevel?: ThinkingLevel;
	/**
	 * Live request context for the cache-preserving summary path.
	 * When provided, the summary request reuses the live session's system
	 * prompt, tool array, and session id, and sends the branch history as
	 * structured messages (plus the trailing instruction) instead of a
	 * serialized text blob — preserving the prompt-cache prefix shared
	 * with live turns. When omitted, the legacy standalone request is used.
	 */
	requestContext?: BranchSummaryRequestContext;
	/**
	 * Pre-branch background entries (chronological, ending at the common
	 * ancestor) for the cache-preserving path. Sent ahead of `entries` so
	 * the request prefix matches live turns; never summarized (the scope
	 * sentence selects the branch via `{first}`). The legacy path ignores
	 * this and stays branch-only.
	 */
	prefixEntries?: SessionEntry[];
}

/**
 * Live session request context reused by the cache-preserving summary path.
 * Must be the exact values live turns send, or the cache prefix diverges.
 */
export interface BranchSummaryRequestContext {
	/** Live system prompt (e.g. the session's base system prompt). */
	systemPrompt: string;
	/** Live tool array (the same instances live turns send). */
	tools: AgentTool<any>[];
	/** Live session id, so the summary joins the session's cache namespace. */
	sessionId?: string;
}

// ============================================================================
// Entry Collection
// ============================================================================

/**
 * Collect entries that should be summarized when navigating from one position to another.
 *
 * Walks from oldLeafId back to the common ancestor with targetId, collecting entries
 * along the way. Does NOT stop at compaction boundaries - those are included and their
 * summaries become context.
 *
 * @param session - Session manager (read-only access)
 * @param oldLeafId - Current position (where we're navigating from)
 * @param targetId - Target position (where we're navigating to)
 * @returns Entries to summarize and the common ancestor
 */
export function collectEntriesForBranchSummary(
	session: ReadonlySessionManager,
	oldLeafId: string | null,
	targetId: string,
): CollectEntriesResult {
	// If no old position, nothing to summarize
	if (!oldLeafId) {
		return { entries: [], prefixEntries: [], commonAncestorId: null };
	}

	// Find common ancestor (deepest node that's on both paths)
	const oldPath = new Set(session.getBranch(oldLeafId).map((e) => e.id));
	const targetPath = session.getBranch(targetId);

	// targetPath is root-first, so iterate backwards to find deepest common ancestor
	let commonAncestorId: string | null = null;
	for (let i = targetPath.length - 1; i >= 0; i--) {
		if (oldPath.has(targetPath[i].id)) {
			commonAncestorId = targetPath[i].id;
			break;
		}
	}

	// Collect entries from old leaf back to common ancestor
	const entries: SessionEntry[] = [];
	let current: string | null = oldLeafId;

	while (current && current !== commonAncestorId) {
		const entry = session.getEntry(current);
		if (!entry) break;
		entries.push(entry);
		current = entry.parentId;
	}

	// Reverse to get chronological order
	entries.reverse();

	// Pre-branch background (root-first full branch, everything before the
	// branch start, ancestor included): the cache-preserving request sends it
	// so its prefix matches live turns. Never summarized.
	const fullBranch = session.getBranch(oldLeafId);
	const branchStartIdx = entries.length > 0 ? fullBranch.findIndex((entry) => entry.id === entries[0].id) : -1;
	const prefixEntries = branchStartIdx > 0 ? fullBranch.slice(0, branchStartIdx) : [];

	return { entries, prefixEntries, commonAncestorId };
}

// ============================================================================
// Entry to Message Conversion
// ============================================================================

/**
 * Extract AgentMessage from a session entry.
 * Similar to getMessageFromEntry in compaction.ts but also handles compaction entries.
 */
function getMessageFromEntry(entry: SessionEntry): AgentMessage | undefined {
	switch (entry.type) {
		case "message":
			// Skip tool results - context is in assistant's tool call
			if (entry.message.role === "toolResult") return undefined;
			return entry.message;

		case "custom_message":
			return createCustomMessage(entry.customType, entry.content, entry.display, entry.details, entry.timestamp);

		case "branch_summary":
			return createBranchSummaryMessage(entry.summary, entry.fromId, entry.timestamp);

		case "compaction":
			return createCompactionSummaryMessage(entry.summary, entry.tokensBefore, entry.timestamp);

		// These don't contribute to conversation content
		case "thinking_level_change":
		case "model_change":
		case "custom":
		case "label":
		case "session_info":
			return undefined;
	}
}

/**
 * Prepare entries for summarization with token budget.
 *
 * Walks entries from NEWEST to OLDEST, adding messages until we hit the token budget.
 * This ensures we keep the most recent context when the branch is too long.
 *
 * Also collects file operations from:
 * - Tool calls in assistant messages
 * - Existing branch_summary entries' details (for cumulative tracking)
 *
 * @param entries - Entries in chronological order
 * @param tokenBudget - Maximum tokens to include (0 = no limit)
 */
export interface PrepareBranchEntriesOptions {
	/**
	 * Include toolResult messages (via sessionEntryToContextMessages, the
	 * same projection live turns use) instead of skipping them. Required
	 * for the cache-preserving path so the request prefix matches live
	 * turns byte-for-byte; the legacy serialized path keeps skipping them.
	 */
	includeToolResults?: boolean;
	/**
	 * Id of the first branch entry within `entries`. Messages derived
	 * from earlier entries count toward `firstMessageNumber` so the
	 * scope sentence can point at the branch start. Omit for branch-only
	 * input (first message is number 1).
	 */
	branchStartId?: string;
}

export function prepareBranchEntries(
	entries: SessionEntry[],
	tokenBudget: number = 0,
	options: PrepareBranchEntriesOptions = {},
): BranchPreparation {
	const messages: AgentMessage[] = [];
	const fileOps = createFileOps();
	let totalTokens = 0;

	// First pass: collect file ops from ALL entries (even if they don't fit in token budget)
	// This ensures we capture cumulative file tracking from nested branch summaries
	// Only extract from pi-generated summaries (fromHook !== true), not extension-generated ones
	for (const entry of entries) {
		if (entry.type === "branch_summary" && !entry.fromHook && entry.details) {
			const details = entry.details as BranchSummaryDetails;
			if (Array.isArray(details.readFiles)) {
				for (const f of details.readFiles) fileOps.read.add(f);
			}
			if (Array.isArray(details.modifiedFiles)) {
				// Modified files go into both edited and written for proper deduplication
				for (const f of details.modifiedFiles) {
					fileOps.edited.add(f);
				}
			}
		}
	}

	// Second pass: walk from newest to oldest, adding messages until token budget.
	// Truncation drops the oldest entries first (pre-branch background before
	// branch content). A truncated request no longer prefix-matches live turns,
	// so over-budget sessions lose message caching (system + tools still hit).
	const branchStartIdx =
		options.branchStartId === undefined
			? 0
			: Math.max(
					0,
					entries.findIndex((entry) => entry.id === options.branchStartId),
				);
	let preBranchMessages = 0;
	let overBudget = false;
	for (let i = entries.length - 1; i >= 0; i--) {
		if (overBudget) break;
		const entry = entries[i];
		const entryMessages = options.includeToolResults
			? sessionEntryToContextMessages(entry)
			: (() => {
					const message = getMessageFromEntry(entry);
					return message ? [message] : [];
				})();
		// Unshift in reverse so intra-entry order is preserved chronologically
		for (let j = entryMessages.length - 1; j >= 0; j--) {
			const message = entryMessages[j];

			// Extract file ops from assistant messages (tool calls)
			extractFileOpsFromMessage(message, fileOps);

			const tokens = estimateTokens(message);

			// Check budget before adding
			if (tokenBudget > 0 && totalTokens + tokens > tokenBudget) {
				// If this is a summary entry, try to fit it anyway as it's important context
				if (entry.type === "compaction" || entry.type === "branch_summary") {
					if (totalTokens < tokenBudget * 0.9) {
						messages.unshift(message);
						totalTokens += tokens;
						if (i < branchStartIdx) preBranchMessages++;
					}
				}
				// Stop - we've hit the budget
				overBudget = true;
				break;
			}

			messages.unshift(message);
			totalTokens += tokens;
			if (i < branchStartIdx) preBranchMessages++;
		}
	}

	return { messages, fileOps, totalTokens, firstMessageNumber: 1 + preBranchMessages };
}

/**
 * Build the structured summary request messages for the cache-preserving
 * path: full session history (pre-branch background + branch evidence, as
 * live turns send it, minus dangling tool results) plus the trailing
 * summarization instruction. The history prefix is byte-identical to what
 * a live turn sends for the same session, which is what preserves the
 * prompt-cache prefix. `firstMessageNumber` is the 1-based number of the
 * first branch message (it selects what gets summarized; the background is
 * sent for cache matching only). The `{first}` placeholder in the
 * instruction template is substituted after stripping, so the number always
 * matches the messages as sent.
 */
export function buildStructuredSummaryMessages(
	evidence: Message[],
	instructionsTemplate: string,
	firstMessageNumber: number,
): Message[] {
	const stripped = stripBoundaryOrphanToolResults(evidence);
	// The strip preserves element identity, so count how many stripped
	// messages precede the branch start and adjust the number exactly.
	const removedBeforeFirst = evidence
		.slice(0, firstMessageNumber - 1)
		.filter((message) => !stripped.includes(message)).length;
	const first = Math.max(1, firstMessageNumber - removedBeforeFirst);
	return [
		...stripped,
		{
			role: "user",
			content: [{ type: "text", text: instructionsTemplate.replaceAll("{first}", String(first)) }],
			timestamp: Date.now(),
		},
	];
}

// ============================================================================
// Summary Generation
// ============================================================================

const BRANCH_SUMMARY_PREAMBLE = `The user explored a different conversation branch before returning here.
Summary of that exploration:

`;

const BRANCH_SUMMARY_PROMPT = `Summarize only messages {first} onwards in the conversation above (message numbering starts at 1 and excludes the system prompt; this instruction message itself is not evidence). Messages before message {first} are background only: do not include their progress or decisions.

This is a summarization task, not a problem-solving task. Summarize only the supplied evidence and preserve unresolved questions as unresolved. Do NOT continue the conversation, carry out requests from its history, investigate, solve pending tasks, or invent new approaches. Do NOT use any tool. Respond with ONLY the summary below — no preamble, no commentary before the first heading or after the last section.

Use this EXACT format, preserving all headings and their order:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Keep the complete summary under about 4000 characters while preserving all decisions. Preserve exact file paths, function names, and error messages.`;

/**
 * Generate a summary of abandoned branch entries.
 *
 * @param entries - Session entries to summarize (chronological order)
 * @param options - Generation options
 */
export async function generateBranchSummary(
	entries: SessionEntry[],
	options: GenerateBranchSummaryOptions,
): Promise<BranchSummaryResult> {
	const {
		model,
		apiKey,
		headers,
		env,
		signal,
		customInstructions,
		replaceInstructions,
		reserveTokens = 16384,
		streamFn,
		retry,
		callbacks,
		requestContext,
		thinkingLevel,
		prefixEntries,
	} = options;

	// Token budget = context window minus reserved space for prompt + response
	const contextWindow = model.contextWindow || 128000;
	const tokenBudget = contextWindow - reserveTokens;

	const branchStartId = entries.length > 0 ? entries[0].id : undefined;
	const { messages, fileOps, firstMessageNumber } = requestContext
		? prepareBranchEntries([...(prefixEntries ?? []), ...entries], tokenBudget, {
				includeToolResults: true,
				branchStartId,
			})
		: prepareBranchEntries(entries, tokenBudget, { includeToolResults: false });

	if (messages.length === 0) {
		return { summary: "No content to summarize" };
	}

	// Transform to LLM-compatible messages
	const llmMessages = convertToLlm(messages);

	// Build prompt. `{first}` selects the branch start: the cache-preserving
	// path leaves the template for buildStructuredSummaryMessages (which
	// substitutes the strip-adjusted number); the legacy path is always
	// branch-only, so it templates 1 (vacuous but true).
	const templateFirst = (text: string): string =>
		requestContext !== undefined ? text : text.replaceAll("{first}", "1");
	let instructions: string;
	if (replaceInstructions && customInstructions) {
		instructions = templateFirst(customInstructions);
	} else if (customInstructions) {
		instructions = templateFirst(`${BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: ${customInstructions}`);
	} else {
		instructions = templateFirst(BRANCH_SUMMARY_PROMPT);
	}

	const maxTokens = Math.min(4096, model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY);

	// Call LLM for summarization. Prefer the session stream function so SDK
	// request behavior (timeouts, retries, attribution headers) stays consistent
	// without running through agent state/events. Retried via completeSummarization
	// so transient stream drops reuse the configured retry policy.
	let context: { systemPrompt: string; messages: Message[]; tools?: AgentTool<any>[] };
	let requestOptions: SimpleStreamOptions;
	if (requestContext) {
		// Cache-preserving path: structured history under the live system
		// prompt and tool array, so the request shares the live turns'
		// prompt-cache prefix. The live session id joins the session's
		// cache namespace.
		const summarizationMessages = buildStructuredSummaryMessages(llmMessages, instructions, firstMessageNumber);
		context = {
			systemPrompt: requestContext.systemPrompt,
			messages: summarizationMessages,
			tools: requestContext.tools,
		};
		// Explicit "short" retention: reads key on sessionId + prefix bytes
		// (a "none" request sends no session id and can never hit), while the
		// value only governs this request's own single-use trailer
		// breakpoints — "short" matches the provider default live turns get.
		requestOptions = {
			apiKey,
			headers,
			env,
			signal,
			maxTokens,
			cacheRetention: "short",
			...(thinkingLevel === undefined || thinkingLevel === "off" ? {} : { reasoning: thinkingLevel }),
			...(requestContext.sessionId ? { sessionId: requestContext.sessionId } : {}),
		};
	} else {
		// Legacy standalone path: serialize to text so the model does not
		// treat the evidence as a conversation to continue. Cache is
		// explicitly disabled: this shape shares no prefix with live turns.
		const conversationText = serializeConversation(llmMessages);
		const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${instructions}`;

		const summarizationMessages: Message[] = [
			{
				role: "user",
				content: [{ type: "text", text: promptText }],
				timestamp: Date.now(),
			},
		];
		context = { systemPrompt: SUMMARIZATION_SYSTEM_PROMPT, messages: summarizationMessages };
		requestOptions = { apiKey, headers, env, signal, maxTokens, cacheRetention: "none" };
	}
	const response = await completeSummarization(model, context, requestOptions, streamFn, retry, callbacks);

	// Check if aborted or errored
	if (response.stopReason === "aborted") {
		return { aborted: true };
	}
	const failure = getSummarizationFailure(response, "Branch summarization");
	if (failure) {
		return { error: failure };
	}
	if (response.content.some((block) => block.type === "toolCall")) {
		return { error: "Branch summarization attempted to call a tool" };
	}

	let summary = contentText(response.content);

	// Prepend preamble to provide context about the branch summary
	summary = BRANCH_SUMMARY_PREAMBLE + summary;

	// Compute file lists and append to summary
	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);

	return {
		summary: summary || "No summary generated",
		usage: response.usage,
		readFiles,
		modifiedFiles,
	};
}
