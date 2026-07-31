/**
 * Pi Notify Extension
 *
 * Sends a native terminal notification when the Pi agent is done and waiting
 * for input. Supports multiple terminal protocols:
 * - OSC 777: Ghostty, iTerm2, WezTerm, rxvt-unicode
 * - OSC 99: Kitty
 * - Windows toast: Windows Terminal (WSL)
 *
 * Background-work protocol
 * ------------------------
 * `agent_settled` only covers Pi's own automatic continuations (retries,
 * compaction retries, queued follow-ups). Work scheduled by *other*
 * extensions is invisible to Pi core: a timer extension holding a
 * `setTimeout`, a heartbeat, or a background subagent will wake the agent
 * with a new message at some future wall-clock time, after `agent_settled`
 * has already fired.
 *
 * Extensions that schedule such work can advertise it through a shared
 * registry so completion notifiers like this one wait for them:
 *
 *   // In the async extension (copy the acquire helper or read the same key):
 *   const done = acquireBackgroundWork("my-timer-extension");
 *   setTimeout(() => {
 *     wakeAgent();
 *     done(); // release the token once the work is handed off
 *   }, 60_000);
 *
 * The registry lives on `globalThis` behind `Symbol.for()`. Pi's loader
 * evaluates each extension with its own module cache, so a module-level
 * singleton in a shared file would NOT be shared — but `Symbol.for()` keys
 * are process-global by spec, so every extension that touches this key gets
 * the same registry. The protocol is the key plus the token shape, not this
 * file: any extension can participate without importing from here.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const REGISTRY_KEY = Symbol.for("pi:background-work");

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
	const template = `[${type}.ToastTemplateType]::ToastText01`;
	const toast = `[${type}.ToastNotification]::new($xml)`;
	return [
		`${mgr} > $null`,
		`$xml = [${type}.ToastNotificationManager]::GetTemplateContent(${template})`,
		`$xml.GetElementsByTagName('text')[0].AppendChild($xml.CreateTextNode('${body}')) > $null`,
		`[${type}.ToastNotificationManager]::CreateToastNotifier('${title}').Show(${toast})`,
	].join("; ");
}

function notifyOSC777(title: string, body: string): void {
	process.stdout.write(`\x1b]777;notify;${title};${body}\x07`);
}

function notifyOSC99(title: string, body: string): void {
	// Kitty OSC 99: i=notification id, d=0 means not done yet, p=body for second part
	process.stdout.write(`\x1b]99;i=1:d=0;${title}\x1b\\`);
	process.stdout.write(`\x1b]99;i=1:p=body;${body}\x1b\\`);
}

function notifyWindows(title: string, body: string): void {
	const { execFile } = require("child_process");
	execFile("powershell.exe", ["-NoProfile", "-Command", windowsToastScript(title, body)]);
}

function notify(title: string, body: string): void {
	if (process.env.WT_SESSION) {
		notifyWindows(title, body);
	} else if (process.env.KITTY_WINDOW_ID) {
		notifyOSC99(title, body);
	} else {
		notifyOSC777(title, body);
	}
}

export default function (pi: ExtensionAPI) {
	const registry = getRegistry();
	let unsubscribeDrain: (() => void) | undefined;
	let disposed = false;

	function cancelDeferred(): void {
		unsubscribeDrain?.();
		unsubscribeDrain = undefined;
	}

	function notifyWhenDrained(ctx: ExtensionContext): void {
		cancelDeferred();
		if (registry.pending() === 0) {
			notify("Pi", "Ready for input");
			return;
		}
		// Background work is still advertised. Wait for the last token to be
		// released instead of notifying now. The listener is one-shot: it
		// unsubscribes itself when it fires so a stale deferral cannot
		// re-fire on a later drain with an outdated context.
		const unsubscribe = registry.onDrain(() => {
			unsubscribe();
			if (unsubscribeDrain === unsubscribe) {
				unsubscribeDrain = undefined;
			}
			// A released token usually coincides with the extension waking the
			// agent (timer fired, background agent completed). Give such a wake
			// a moment to start a new run; if it did, agent_start has already
			// cancelled this path and isIdle() will be false here.
			setTimeout(() => {
				if (!disposed && ctx.isIdle()) {
					notify("Pi", "Ready for input");
				}
			}, 250);
		});
		unsubscribeDrain = unsubscribe;
	}

	// Use `agent_settled` rather than `agent_end`: `agent_end` fires after each
	// low-level run, but Pi may still auto-retry, auto-compact and retry, or
	// continue with queued follow-up messages. `agent_settled` fires only when
	// Pi will not continue running automatically.
	pi.on("agent_settled", async (_event, ctx) => {
		notifyWhenDrained(ctx);
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
