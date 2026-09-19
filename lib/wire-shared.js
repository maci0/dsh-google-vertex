/**
 * Helpers the two publisher wire translators (Anthropic and Gemini) share:
 * tool-argument parsing, system-text collection, and usage-counter merging.
 *
 * @module dsh-google-vertex/wire-shared
 */
/** Parse a model-produced arguments string into the object the provider requires. */
export function toolInput(argumentsJson) {
    try {
        const parsed = JSON.parse(argumentsJson);
        return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
            ? parsed
            : {};
    }
    catch {
        // The harness assembles these strings from the model's own deltas; an
        // unparseable one is a provider-side truncation, and an empty object keeps
        // the turn revisable instead of failing the whole request.
        return {};
    }
}
/** System-role text the request carries, in assembly order. */
export function systemParts(options) {
    const parts = [];
    if (options.system !== undefined && options.system.length > 0)
        parts.push(options.system);
    for (const message of options.messages) {
        if (message.role !== 'system')
            continue;
        for (const block of message.content) {
            if (block.type === 'text' && typeof block.text === 'string' && block.text.length > 0)
                parts.push(block.text);
        }
    }
    return parts;
}
/** A usage counter, ignoring anything the provider sends that is not a number. */
export function count(value) {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
}
/**
 * Read the named counters out of one usage payload, keeping only the ones the
 * provider actually sent.
 * @param raw - the payload's usage member.
 * @param names - the counter names this route reads.
 * @returns only the present, well-formed counters.
 */
export function takeCounters(raw, names) {
    const merged = {};
    if (typeof raw !== 'object' || raw === null)
        return merged;
    const usage = raw;
    for (const name of names) {
        const value = count(usage[name]);
        if (value !== undefined)
            merged[name] = value;
    }
    return merged;
}
