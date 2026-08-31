/**
 * Pi Notify Extension
 *
 * Sends a native terminal notification when the Pi agent is done and waiting
 * for input. Supports multiple terminal protocols:
 * - OSC 777: `777;notify;title;body` — structured title+body, supported by
 *   Ghostty, WezTerm, Foot, VTE-based terminals, Warp, VS Code terminal
 *   (rxvt-unicode origin; terminal-wg spec issue 13 is standardizing it)
 * - OSC 99: Kitty (structured title/body via `p=body`, id, urgency)
 * - OSC 9: iTerm2 (message-only; title and body are concatenated)
 * - Windows toast: Windows PowerShell 5.1 WinRT toast, gated on `WT_SESSION`
 *   (WSL / Windows Terminal). Windows Terminal implements neither OSC 9
 *   notifications (only the `9;4` progress sub-protocol) nor usable OSC 777
 *   (package identity + foreground gating), so the toast is the only
 *   reliable Windows mechanism. Unaffected by `PI_NOTIFY_PROTOCOL`.
 *
 * Protocol selection is per-backend with a user override:
 * `PI_NOTIFY_PROTOCOL = auto | osc9 | osc777 | osc99` (default auto). In
 * auto mode an unknown terminal gets no notification at all — better silent
 * than emitting a sequence the terminal does not understand; forcing a
 * protocol skips detection entirely.
 *
 * Settle pipeline
 * ---------------
 * `agent_settled` only covers Pi's own automatic continuations (retries,
 * compaction retries, queued follow-ups). Work scheduled by *other*
 * extensions is invisible to Pi core: a timer extension holding a
 * `setTimeout`, a heartbeat, or a background subagent will wake the agent
 * with a new message at some future wall-clock time, after `agent_settled`
 * has already fired.
 *
 * Extensions that schedule such work advertise it through a shared registry
 * (see below), and completion notifiers like this one wait for it. On top of
 * the registry, delivery follows two phases:
 *
 *   1. Deferral: on `agent_settled`, wait for the registry to drain, then
 *      hold the notification for a configurable grace period
 *      (`PI_NOTIFY_SETTLE_GRACE_MS`, default 1000, 0 = immediate).
 *   2. Fire-time revalidation: when the grace period elapses, re-check the
 *      FULL settle condition — `ctx.isIdle()` AND `pending() === 0` — and
 *      drop the notification if either no longer holds. `agent_start`
 *      cancels the deferral outright.
 *
 * This is the same shape as herdr's notification pipeline (Working|Blocked →
 * Idle transition, `delay_seconds` grace, re-validate state and agent label
 * at delivery time, cancel on any new state change). The grace period makes
 * rapid re-engagement (auto-retry, queued follow-ups, a timer firing right
 * after the drain) invisible to the user, and the fire-time revalidation is
 * what makes a long grace period safe: correctness is decided when the
 * notification actually fires, not when the settle was detected.
 *
 * No dedup state is needed: `agent_settled` is emitted exactly once per run
 * by construction — the single emit site is the `finally` of
 * `agent-session._runAgentPrompt`, and every run begins with `agent_start`
 * (the runner emits it at the top of `runAgentLoop` / `runAgentLoopContinue`,
 * which cancels any active deferral here).
 *
 * Background-work protocol
 * ------------------------
 * The registry lives on `globalThis` behind `Symbol.for()`. Pi's loader
 * evaluates each extension with its own module cache, so a module-level
 * singleton in a shared file would NOT be shared — but `Symbol.for()` keys
 * are process-global by spec, so every extension that touches this key gets
 * the same registry. The protocol is the key plus the token shape, not this
 * file: any extension can participate without importing from here.
 */

import { execFile } from "node:child_process";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const REGISTRY_KEY = Symbol.for("pi:background-work");

const rawGraceMs = Number(process.env.PI_NOTIFY_SETTLE_GRACE_MS ?? 1000);
const SETTLE_GRACE_MS = Number.isFinite(rawGraceMs) && rawGraceMs >= 0 ? rawGraceMs : 1000;

const rawProtocol = process.env.PI_NOTIFY_PROTOCOL ?? "auto";
const PROTOCOL: "auto" | "osc9" | "osc777" | "osc99" =
	rawProtocol === "osc9" || rawProtocol === "osc777" || rawProtocol === "osc99"
		? rawProtocol
		: "auto";

interface BackgroundWorkRegistry {
	/** Number of background tasks currently advertised as pending. */
	pending(): number;
	/**
	 * Advertise a pending background task. Returns a release function; call it
	 * exactly once when the task completes or is cancelled. Safe to call more
	 * than once (subsequent calls are no-ops).
	 */
	acquire(source: string): () => void;
	/**
	 * Subscribe to the registry reaching zero pending tasks. Returns an
	 * unsubscribe function. Not called while tasks remain pending.
	 */
	onDrain(listener: () => void): () => void;
}

function getRegistry(): BackgroundWorkRegistry {
	const g = globalThis as unknown as Record<symbol, BackgroundWorkRegistry | undefined>;
	let registry = g[REGISTRY_KEY];
	if (!registry) {
		const tasks = new Map<number, string>();
		const drainListeners = new Set<() => void>();
		let nextId = 0;
		registry = {
			pending: () => tasks.size,
			acquire(source: string) {
				const id = ++nextId;
				tasks.set(id, source);
				let released = false;
				return () => {
					if (released) return;
					released = true;
					if (!tasks.delete(id)) return;
					if (tasks.size === 0) {
						for (const listener of [...drainListeners]) listener();
					}
				};
			},
			onDrain(listener: () => void) {
				drainListeners.add(listener);
				return () => {
					drainListeners.delete(listener);
				};
			},
		};
		g[REGISTRY_KEY] = registry;
	}
	return registry;
}

/**
 * Advertise pending background work from any extension. Call the returned
 * release function when the work completes or is cancelled.
 */
export function acquireBackgroundWork(source: string): () => void {
	return getRegistry().acquire(source);
}

function windowsToastScript(title: string, body: string): string {
	const type = "Windows.UI.Notifications";
	const mgr = `[${type}.ToastNotificationManager, ${type}, ContentType = WindowsRuntime]`;
	const template = `[${type}.ToastTemplateType]::ToastText02`;
	const toast = `[${type}.ToastNotification]::new($xml)`;
	// PowerShell single-quote escaping: interpolated content must not break
	// out of the -Command string.
	const safeTitle = title.replaceAll("'", "''");
	const safeBody = body.replaceAll("'", "''");
	return [
		`${mgr} > $null`,
		`$xml = [${type}.ToastNotificationManager]::GetTemplateContent(${template})`,
		`$xml.GetElementsByTagName('text')[0].AppendChild($xml.CreateTextNode('${safeTitle}')) > $null`,
		`$xml.GetElementsByTagName('text')[1].AppendChild($xml.CreateTextNode('${safeBody}')) > $null`,
		// "Windows PowerShell" is a registered AUMID; toasts from unregistered
		// appIds can be dropped silently on Windows 11.
		`[${type}.ToastNotificationManager]::CreateToastNotifier('Windows PowerShell').Show(${toast})`,
	].join("; ");
}

function notifyWindows(title: string, body: string): void {
	// Best-effort delivery: powershell.exe is Windows PowerShell 5.1 — the
	// only edition with the WinRT toast assemblies (pwsh 7+ does not ship
	// them). The callback is required: a callback-less execFile emits an
	// unhandled "error" event and crashes the host process when interop is
	// unavailable (WSL interop disabled, missing powershell.exe).
	execFile("powershell.exe", ["-NoProfile", "-Command", windowsToastScript(title, body)], () => {
		// Ignored: toast delivery is best-effort (interop off, per-app toast
		// settings). There is no OSC fallback under Windows Terminal — it
		// implements neither OSC 9 notifications nor usable OSC 777.
	});
}

// Backend detection follows herdr's terminal_notify.rs; sequence construction
// is protocol-mapped per backend. All OSC paths sanitize control bytes: an
// ESC or BEL in the message would terminate/alter the OSC frame (BEL and ST
// are both terminators), so unfiltered content could inject arbitrary escape
// sequences; newlines are flattened because notifications are single-line.
type TerminalBackend = "ghostty" | "iterm2" | "kitty" | "wezterm" | "vscode";

function detectBackend(): TerminalBackend | null {
	switch (process.env.TERM_PROGRAM) {
		case "ghostty":
			return "ghostty";
		case "iTerm.app":
			return "iterm2";
		case "WezTerm":
			return "wezterm";
		case "vscode":
			// VS Code integrated terminal (common WSL host): renders OSC 777
			// with the terminal-notification extension or recent builds;
			// without either the sequence is swallowed harmlessly.
			return "vscode";
	}
	if (process.env.KITTY_WINDOW_ID) {
		return "kitty";
	}
	switch (process.env.TERM) {
		case "xterm-ghostty":
			return "ghostty";
		case "xterm-kitty":
			return "kitty";
		default:
			return process.env.TERM?.includes("wezterm") ? "wezterm" : null;
	}
}

function protocolForBackend(backend: TerminalBackend): "osc9" | "osc777" | "osc99" {
	switch (backend) {
		case "kitty":
			return "osc99";
		case "iterm2":
			return "osc9"; // iTerm2's documented native protocol
		case "ghostty":
		case "wezterm":
		case "vscode":
			// OSC 777 is the terminal-wg direction and carries a real
			// title/body split.
			return "osc777";
	}
}

function sanitizeText(text: string): string {
	let out = "";
	for (const ch of text) {
		if (ch === "\u001b" || ch === "\u0007" || ch === "\u009c") continue;
		out += ch === "\n" || ch === "\r" || ch === "\t" ? " " : ch;
	}
	return out;
}

function buildOSC9(title: string, body: string): string {
	const message = sanitizeText(body ? `${title}: ${body}` : title);
	return `\x1b]9;${message}\x1b\\`;
}

function buildOSC777(title: string, body: string): string {
	return `\x1b]777;notify;${sanitizeText(title)};${sanitizeText(body)}\x1b\\`;
}

function buildOSC99(title: string, body: string): string {
	// Kitty OSC 99: i=notification id, d=0 means live, p=body for second part
	let sequence = `\x1b]99;i=1:d=0;${sanitizeText(title)}\x1b\\`;
	if (body) {
		sequence += `\x1b]99;i=1:p=body;${sanitizeText(body)}\x1b\\`;
	}
	return sequence;
}

function wrapTmuxPassthrough(sequence: string): string {
	// DCS passthrough: every ESC inside the payload must be doubled. Requires
	// `set -g allow-passthrough on` in tmux.conf; without it the DCS is
	// dropped harmlessly.
	return `\x1bPtmux;${sequence.replaceAll("\u001b", "\u001b\u001b")}\x1b\\`;
}

function notifyTerminal(title: string, body: string): void {
	let protocol: "auto" | "osc9" | "osc777" | "osc99" = PROTOCOL;
	if (protocol === "auto") {
		const backend = detectBackend();
		if (!backend) {
			return;
		}
		protocol = protocolForBackend(backend);
	}
	let sequence =
		protocol === "osc99"
			? buildOSC99(title, body)
			: protocol === "osc777"
				? buildOSC777(title, body)
				: buildOSC9(title, body);
	if (process.env.TMUX) {
		sequence = wrapTmuxPassthrough(sequence);
	}
	process.stdout.write(sequence);
}

function notify(title: string, body: string): void {
	if (process.env.WT_SESSION) {
		notifyWindows(title, body);
	} else {
		notifyTerminal(title, body);
	}
}

export default function (pi: ExtensionAPI) {
	const registry = getRegistry();
	let unsubscribeDrain: (() => void) | undefined;
	let fireTimer: ReturnType<typeof setTimeout> | undefined;
	let disposed = false;

	function cancelDeferred(): void {
		unsubscribeDrain?.();
		unsubscribeDrain = undefined;
		if (fireTimer !== undefined) {
			clearTimeout(fireTimer);
			fireTimer = undefined;
		}
	}

	function fireIfStillSettled(ctx: ExtensionContext): void {
		fireTimer = undefined;
		if (disposed) {
			return;
		}
		// Fire-time revalidation (herdr re-checks state + agent label at
		// delivery time; the closest pi equivalents are idle state and the
		// registry). An agent_start in the window already cancelled this
		// deferral; a token acquired since the drain is caught here.
		if (!ctx.isIdle() || registry.pending() > 0) {
			return;
		}
		notify("Pi", "Ready for input");
	}

	function notifyWhenSettled(ctx: ExtensionContext): void {
		const scheduleFire = () => {
			fireTimer = setTimeout(() => fireIfStillSettled(ctx), SETTLE_GRACE_MS);
		};
		if (registry.pending() > 0) {
			// Background work is still advertised. Wait for the last token to
			// be released, then start the grace period. The listener is
			// one-shot: it unsubscribes itself so a stale deferral cannot
			// re-fire on a later drain with an outdated context.
			const unsubscribe = registry.onDrain(() => {
				unsubscribe();
				if (unsubscribeDrain === unsubscribe) {
					unsubscribeDrain = undefined;
				}
				scheduleFire();
			});
			unsubscribeDrain = unsubscribe;
		} else {
			scheduleFire();
		}
	}

	// Use `agent_settled` rather than `agent_end`: `agent_end` fires after each
	// low-level run, but Pi may still auto-retry, auto-compact and retry, or
	// continue with queued follow-up messages. `agent_settled` fires only when
	// Pi will not continue running automatically.
	pi.on("agent_settled", async (_event, ctx) => {
		notifyWhenSettled(ctx);
	});

	// A new run starting (e.g. a timer waking the agent) invalidates any
	// deferred notification; the run's own agent_settled will re-evaluate.
	pi.on("agent_start", async () => {
		cancelDeferred();
	});

	pi.on("session_shutdown", async () => {
		disposed = true;
		cancelDeferred();
	});
}