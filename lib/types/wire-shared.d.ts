/**
 * Helpers the two publisher wire translators (Anthropic and Gemini) share:
 * tool-argument parsing, system-text collection, and usage-counter merging.
 *
 * @module dsh-google-vertex/wire-shared
 */
import type { GenerateOptions } from './host.ts';
/** Parse a model-produced arguments string into the object the provider requires. */
export declare function toolInput(argumentsJson: string): Record<string, unknown>;
/** System-role text the request carries, in assembly order. */
export declare function systemParts(options: GenerateOptions): string[];
/**
 * Read the named counters out of one usage payload, keeping only the ones the
 * provider actually sent.
 * @param raw - the payload's usage member.
 * @param names - the counter names this route reads.
 * @returns only the present, well-formed counters.
 */
export declare function takeCounters(raw: unknown, names: readonly string[]): Record<string, number>;
