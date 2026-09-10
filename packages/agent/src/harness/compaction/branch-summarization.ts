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
}

/** Entries selected for branch summarization. */
export interface CollectEntriesResult {
	/** Entries to summarize in chronological order. */
	entries: Entry[];
	/** Deepest common ancestor between the previous tip and target entry. */
	commonAncestorId: string | null;
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
		return { entries: [], commonAncestorId: null };
	}
	const oldPath = new Set((await branch.findEntries({ start: oldTipId }, context)).map((entry) => entry.id));
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

	return { entries, commonAncestorId };
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
					}
				}
				overBudget = true;
				break;
			}

			messages.unshift(message);
			totalTokens += tokens;
		}
	}

	return { messages, fileOps, totalTokens };
}

/**
 * Build the structured summary request messages for the cache-preserving
 * path: branch evidence (as live turns send it, minus boundary-orphan tool
 * results) plus the trailing summarization instruction. The evidence prefix
 * is byte-identical to what a live turn would send for the same history,
 * which is what preserves the prompt-cache prefix.
 */
export function buildStructuredSummaryMessages(evidence: Message[], instructions: string): Message[] {
	return [
		...stripBoundaryOrphanToolResults(evidence),
		{
			role: "user",
			content: [{ type: "text", text: instructions }],
			timestamp: Date.now(),
		},
	];
}

const BRANCH_SUMMARY_PREAMBLE = `The user explored a different conversation branch before returning here.
Summary of that exploration:

`;

const BRANCH_SUMMARY_PROMPT = `Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

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
	const { messages, fileOps } = preparation;
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
			messages: buildStructuredSummaryMessages(llmMessages, instructions),
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
		const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${instructions}`;

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
