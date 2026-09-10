import {
	type Api,
	contentText,
	type Message,
	type Model,
	type Models,
	type RetryCallbacks,
	type RetryPolicy,
	type SimpleStreamOptions,
	type Tool,
	type Usage,
} from "@earendil-works/pi-ai";
import type { AgentMessage, ThinkingLevel } from "../../types.ts";
import type { Context } from "../context.ts";
import { convertToLlm, createBranchSummaryMessage, createCompactionSummaryMessage } from "../messages.ts";
import { sessionEntryToContextMessages } from "../session/context.ts";
import type { Branch, Entry, Session } from "../session/index.ts";
import { BranchSummaryError, err, ok, type Result } from "../types.ts";
import type { BranchSummaryCacheMiss } from "./cache-miss.ts";
import {
	completeSimpleWithRetries,
	createSummaryRequestOptions,
	estimateTokens,
	SUMMARIZATION_SYSTEM_PROMPT,
	type SummaryRequest,
} from "./compaction.ts";
import {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	type FileOperations,
	formatFileOperations,
	serializeConversation,
	stripBoundaryOrphanToolResults,
} from "./utils.ts";

/** Generated branch summary data ready to be persisted as a branch-summary entry. */
export interface BranchSummaryResult {
	summary: string;
	usage?: Usage;
	readFiles: string[];
	modifiedFiles: string[];
	/**
	 * Measured prompt-cache miss paid by the summary request itself, if any.
	 * Attached by the structural drive from the measured response usage;
	 * display metadata only, never billed-token accounting.
	 */
	cacheMiss?: BranchSummaryCacheMiss;
}

/** File-operation details stored on generated branch summary entries. */
export interface BranchSummaryDetails {
	/** Files read while exploring the summarized branch. */
	readFiles: string[];
	/** Files modified while exploring the summarized branch. */
	modifiedFiles: string[];
}

export type { FileOperations } from "./utils.ts";

/** Prepared branch content for summarization. */
export interface BranchPreparation {
	/** Messages selected for the branch summary. */
	messages: AgentMessage[];
	/** File operations extracted from the branch. */
	fileOps: FileOperations;
	/** Estimated token count for selected messages. */
	totalTokens: number;
	/**
	 * 1-based number of the first branch message within `messages`
	 * (message numbering starts at 1 and excludes the system prompt).
	 * Pre-branch background messages precede it; they are sent so the
	 * request shares the live turns' cache prefix, not to be summarized.
	 */
	firstMessageNumber: number;
}

/** Entries selected for branch summarization. */
export interface CollectEntriesResult {
	/** Entries to summarize in chronological order. */
	entries: Entry[];
	/** Deepest common ancestor between the previous tip and target entry. */
	commonAncestorId: string | null;
	/**
	 * Pre-branch background entries (chronological, ancestor included).
	 * Sent with the cache-preserving request so its prefix matches live
	 * turns; never summarized.
	 */
	prefixEntries: Entry[];
}

/** Options for generating a branch summary. */
export interface GenerateBranchSummaryOptions {
	/** Provider collection the summarization request goes through; owns auth resolution. */
	models: Models;
	/** Model used for summarization. */
	model: Model<Api>;
	/** Optional instructions appended to or replacing the default prompt. */
	customInstructions?: string;
	/** Replace the default prompt with custom instructions instead of appending them. */
	replaceInstructions?: boolean;
	/** Tokens reserved for prompt and model output. Defaults to 16384. */
	reserveTokens?: number;
	/** Optional retry policy for transient summarization errors. */
	retry?: RetryPolicy;
	/** Optional callbacks for retry reporting. */
	callbacks?: RetryCallbacks;
}

/** Collect entries that should be summarized before navigating to a different session tree entry. */
export async function collectEntriesForBranchSummary(
	branch: Pick<Branch, "findEntries">,
	session: Pick<Session, "getEntry">,
	oldTipId: string | null,
	targetId: string,
	context: Context,
): Promise<CollectEntriesResult> {
	if (!oldTipId) {
		return { entries: [], prefixEntries: [], commonAncestorId: null };
	}
	// Tip-first chain to the session root.
	const oldChain = await branch.findEntries({ start: oldTipId }, context);
	const oldPath = new Set(oldChain.map((entry) => entry.id));
	const targetPath = await branch.findEntries({ start: targetId }, context);
	let commonAncestorId: string | null = null;
	for (const entry of targetPath) {
		if (oldPath.has(entry.id)) {
			commonAncestorId = entry.id;
			break;
		}
	}
	const entries: Entry[] = [];
	let current: string | null = oldTipId;

	while (current && current !== commonAncestorId) {
		const entry = await session.getEntry(current, context);
		if (!entry) throw new Error(`Corrupt session: entry ${current} not found`);
		entries.push(entry);
		current = entry.parentId;
	}
	entries.reverse();

	// Pre-branch background (everything older than the branch start,
	// ancestor included), chronological. Sent with the cache-preserving
	// request so its prefix matches live turns; never summarized.
	const ancestorIdx = oldChain.findIndex((entry) => entry.id === commonAncestorId);
	const prefixEntries = ancestorIdx >= 0 ? oldChain.slice(ancestorIdx).reverse() : [];

	return { entries, prefixEntries, commonAncestorId };
}
function getMessageFromEntry(entry: Entry): AgentMessage | undefined {
	switch (entry.type) {
		case "message":
			if (entry.message.role === "toolResult") return undefined;
			return entry.message;

		case "branch_summary":
			return createBranchSummaryMessage(entry.summary, entry.fromId, entry.timestamp);

		case "compaction":
			return createCompactionSummaryMessage(entry.summary, entry.tokensBefore, entry.timestamp);
		case "custom":
			return undefined;
	}
}

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

/** Prepare branch entries for summarization within an optional token budget. */
export function prepareBranchEntries(
	entries: Entry[],
	tokenBudget: number = 0,
	options: PrepareBranchEntriesOptions = {},
): BranchPreparation {
	const messages: AgentMessage[] = [];
	const fileOps = createFileOps();
	let totalTokens = 0;
	for (const entry of entries) {
		if (
			entry.type !== "branch_summary" ||
			typeof entry.details !== "object" ||
			entry.details === null ||
			Array.isArray(entry.details)
		) {
			continue;
		}
		if (Array.isArray(entry.details.readFiles)) {
			for (const path of entry.details.readFiles) {
				if (typeof path === "string") fileOps.read.add(path);
			}
		}
		if (Array.isArray(entry.details.modifiedFiles)) {
			for (const path of entry.details.modifiedFiles) {
				if (typeof path === "string") fileOps.edited.add(path);
			}
		}
	}
	// Walk newest to oldest; truncation drops the oldest first (pre-branch
	// background before branch content). A truncated request no longer
	// prefix-matches live turns, so over-budget sessions lose message
	// caching (system + tools still hit).
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
			extractFileOpsFromMessage(message, fileOps);

			const tokens = estimateTokens(message);
			if (tokenBudget > 0 && totalTokens + tokens > tokenBudget) {
				if (entry.type === "compaction" || entry.type === "branch_summary") {
					if (totalTokens < tokenBudget * 0.9) {
						messages.unshift(message);
						totalTokens += tokens;
						if (i < branchStartIdx) preBranchMessages++;
					}
				}
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

/** Generate a summary for abandoned branch entries. */
export function generateBranchSummary(
	entries: Entry[],
	options: GenerateBranchSummaryOptions,
	context: Context,
): Promise<Result<BranchSummaryResult, BranchSummaryError>> {
	const { models, model, customInstructions, replaceInstructions, reserveTokens = 16384, retry, callbacks } = options;
	const contextWindow = model.contextWindow || 128000;
	const preparation = prepareBranchEntries(entries, contextWindow - reserveTokens);
	return generateBranchSummaryWithRequest(
		preparation,
		{ customInstructions, replaceInstructions },
		(aiContext, requestOptions, requestContext) =>
			completeSimpleWithRetries(models, model, aiContext, requestOptions, retry, callbacks, requestContext),
		context,
	);
}

export interface PreparedBranchSummaryOptions {
	customInstructions?: string;
	replaceInstructions?: boolean;
	/**
	 * Live thinking level, forwarded as the request reasoning effort on the
	 * cache-preserving path only (same rationale as the coding-agent path:
	 * the provider keys cache on effort). The legacy path is unchanged.
	 */
	thinkingLevel?: ThinkingLevel;
	/**
	 * Live request context for the cache-preserving summary path.
	 * When provided, the summary request reuses the live lane's system
	 * prompt, tool array, and session id, and sends the branch history as
	 * structured messages (plus the trailing instruction) instead of a
	 * serialized text blob — preserving the prompt-cache prefix shared
	 * with live turns. When omitted, the legacy standalone request is used.
	 */
	requestContext?: BranchSummaryRequestContext;
}

/**
 * Live lane request context reused by the cache-preserving summary path.
 * Must be the exact values live turns send, or the cache prefix diverges.
 */
export interface BranchSummaryRequestContext {
	/** Live system prompt. */
	systemPrompt: string;
	/** Live tool array (the same instances live turns send). */
	tools: Tool[];
	/** Live session id, so the summary joins the session's cache namespace. */
	sessionId?: string;
}

/** Generate a prepared branch summary through a caller-owned one-request boundary. */
export async function generateBranchSummaryWithRequest(
	preparation: BranchPreparation,
	options: PreparedBranchSummaryOptions,
	request: SummaryRequest,
	context: Context,
): Promise<Result<BranchSummaryResult, BranchSummaryError>> {
	const { customInstructions, replaceInstructions, requestContext, thinkingLevel } = options;
	const { messages, fileOps, firstMessageNumber } = preparation;
	if (messages.length === 0) {
		return ok({ summary: "No content to summarize", readFiles: [], modifiedFiles: [] });
	}
	const llmMessages = convertToLlm(messages);
	let instructions: string;
	if (replaceInstructions && customInstructions) {
		instructions = customInstructions;
	} else if (customInstructions) {
		instructions = `${BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: ${customInstructions}`;
	} else {
		instructions = BRANCH_SUMMARY_PROMPT;
	}
	// NOTE: the 2048 output cap is unchanged by the cache-preserving path.
	const summaryRequestOptions: SimpleStreamOptions = { maxTokens: 2048 };
	let aiContext: { systemPrompt: string; messages: Message[]; tools?: Tool[] };
	if (requestContext) {
		// Cache-preserving path: structured history under the live system
		// prompt and tool array. Cache retention stays at the live default
		// and the live session id joins the session's cache namespace.
		aiContext = {
			systemPrompt: requestContext.systemPrompt,
			messages: buildStructuredSummaryMessages(llmMessages, instructions, firstMessageNumber),
			tools: requestContext.tools,
		};
		if (requestContext.sessionId) summaryRequestOptions.sessionId = requestContext.sessionId;
		if (thinkingLevel !== undefined && thinkingLevel !== "off") summaryRequestOptions.reasoning = thinkingLevel;
		// Explicit "short" retention: reads key on sessionId + prefix bytes
		// (a "none" request sends no session id and can never hit), while the
		// value only governs this request's own single-use trailer
		// breakpoints — "short" matches the provider default live turns get.
		summaryRequestOptions.cacheRetention = "short";
	} else {
		// Legacy standalone path: serialize to text so the model does not
		// treat the evidence as a conversation to continue. Cache is
		// explicitly disabled: this shape shares no prefix with live turns.
		const conversationText = serializeConversation(llmMessages);
		// Legacy path is always branch-only: template 1 (vacuous but true).
		const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${instructions.replaceAll("{first}", String(firstMessageNumber))}`;

		const summarizationMessages: Message[] = [
			{
				role: "user",
				content: [{ type: "text", text: promptText }],
				timestamp: Date.now(),
			},
		];
		aiContext = { systemPrompt: SUMMARIZATION_SYSTEM_PROMPT, messages: summarizationMessages };
		summaryRequestOptions.cacheRetention = "none";
	}
	const response = await request(aiContext, createSummaryRequestOptions(summaryRequestOptions, context), context);
	if (response.stopReason === "aborted") {
		return err(new BranchSummaryError("aborted", response.errorMessage || "Branch summary aborted"));
	}
	if (response.stopReason === "error") {
		return err(
			new BranchSummaryError(
				"summarization_failed",
				`Branch summary failed: ${response.errorMessage || "Unknown error"}`,
			),
		);
	}

	let summary = contentText(response.content);
	summary = BRANCH_SUMMARY_PREAMBLE + summary;
	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);

	return ok({
		summary: summary || "No summary generated",
		usage: response.usage,
		readFiles,
		modifiedFiles,
	});
}
