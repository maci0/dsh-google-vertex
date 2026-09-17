/**
 * Wire translation for Google's own Gemini models on Vertex AI: the request
 * body `publishers/google` accepts, the SSE response shape, and — the part that
 * makes tool calling work at all on Gemini 3 — the thought signature that has to
 * travel back with every replayed function call.
 *
 * Everything here is pure: the adapter feeds it bytes and yields the chunks it
 * returns, so the protocol is testable without a harness or a network.
 *
 * @module dsh-google-vertex/gemini
 */
import { EMPTY_RESPONSE_CODE, endpointOrigin, failureForEvent, resultText } from './wire.js';
/** Harness code reported when the provider refused on safety grounds. */
export const SAFETY_BLOCKED_CODE = 'SAFETY';
/** Vertex path version this adapter speaks. */
const API_VERSION = 'v1';
/**
 * Gemini context capacity: every current Gemini model serves roughly a million
 * tokens, and the catalog below narrows nothing.
 */
export const DEFAULT_GEMINI_CONTEXT_WINDOW = 1_048_576;
/**
 * Output cap applied when a caller omits one.
 *
 * Vertex's ceiling here is EXCLUSIVE — `maxOutputTokens: 65536` is refused with
 * "supported range is from 1 (inclusive) to 65536 (exclusive)" — so the highest
 * accepted value is one less.
 */
export const DEFAULT_GEMINI_MAX_TOKENS = 65_535;
/**
 * Gemini models the global endpoint serves, in picker order. Ids are the
 * provider's own aliases, so a promoted release needs no edit here.
 */
export const DEFAULT_GEMINI_MODELS = [
    { id: 'gemini-3.5-flash', name: 'Gemini 3.5 Flash (Vertex)' },
    { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro Preview (Vertex)' },
    { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash Preview (Vertex)' },
    { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro (Vertex)' },
    { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash (Vertex)' },
    { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash-Lite (Vertex)' },
];
/**
 * The streaming publisher path for one Gemini model.
 * @param project - Google Cloud project id.
 * @param location - region, or `global`.
 * @param model - publisher model id, e.g. `gemini-3.5-flash`.
 * @returns the absolute request URL, with the SSE response encoding.
 */
export function geminiEndpointFor(project, location, model) {
    const origin = endpointOrigin(location);
    const path = `/${API_VERSION}/projects/${encodeURIComponent(project)}/locations/${encodeURIComponent(location)}`
        + `/publishers/google/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`;
    return `${origin}${path}`;
}
/**
 * Build the replay envelope for one finished response.
 * @param model - the model that produced the response.
 * @param blocks - per-block metadata in emitted order.
 * @returns the envelope for the terminal finish chunk.
 */
export function geminiReplayState(model, blocks) {
    return { response: { kind: 'google-vertex-gemini', version: 1, model }, blocks };
}
/** A JSON object, or undefined for anything else. */
function asObject(value) {
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? value : undefined;
}
/**
 * Read back the replay metadata for one assistant message.
 *
 * Anything unexpected — a foreign envelope, another model, a block count that no
 * longer lines up with the content — yields undefined, which degrades to sending
 * the call without its signature rather than throwing: the provider then decides,
 * and a cross-provider history stays replayable.
 * @param message - the assistant message from history.
 * @param model - the model about to be called; signatures do not cross models.
 * @returns index-aligned metadata, or undefined when unusable.
 */
export function readGeminiReplay(message, model) {
    const source = message.source;
    if (source?.kind !== 'model' || source.replayState === undefined)
        return undefined;
    const envelope = asObject(source.replayState);
    const response = asObject(envelope?.['response']);
    if (response?.['kind'] !== 'google-vertex-gemini' || response['version'] !== 1)
        return undefined;
    if (response['model'] !== model)
        return undefined;
    const raw = envelope?.['blocks'];
    if (!Array.isArray(raw) || raw.length !== message.content.length)
        return undefined;
    const blocks = [];
    for (const [index, value] of raw.entries()) {
        const block = asObject(value);
        const type = block?.['type'];
        if (type !== message.content[index]?.type)
            return undefined;
        if (type !== 'text' && type !== 'tool-call')
            return undefined;
        const signature = block?.['thoughtSignature'];
        if (signature !== undefined && typeof signature !== 'string')
            return undefined;
        blocks.push({ type, ...signature === undefined ? {} : { thoughtSignature: signature } });
    }
    return blocks;
}
/** Parse a model-produced arguments string into the object Gemini requires. */
function toolInput(argumentsJson) {
    try {
        const parsed = JSON.parse(argumentsJson);
        return asObject(parsed) ?? {};
    }
    catch {
        return {};
    }
}
/** Every tool-call id in this request, mapped to the tool's name. */
function toolNames(messages) {
    const names = new Map();
    for (const message of messages) {
        for (const block of message.content) {
            if (block.type === 'tool-call' && typeof block.id === 'string')
                names.set(block.id, String(block.name));
        }
    }
    return names;
}
/** System text this request carries, in assembly order. */
function systemText(options) {
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
    return parts.join('\n\n');
}
/** Project one assistant message onto Gemini parts, signatures included. */
function assistantParts(message, model) {
    const replay = readGeminiReplay(message, model);
    const parts = [];
    message.content.forEach((block, index) => {
        const signature = replay?.[index]?.thoughtSignature;
        if (block.type === 'text') {
            if (typeof block.text === 'string' && block.text.length > 0) {
                parts.push({ text: block.text, ...signature === undefined ? {} : { thoughtSignature: signature } });
            }
            return;
        }
        if (block.type === 'tool-call') {
            const call = {
                name: String(block.name),
                args: toolInput(String(block.arguments)),
                ...typeof block.id === 'string' && block.id.length > 0 ? { id: block.id } : {},
            };
            parts.push({ functionCall: call, ...signature === undefined ? {} : { thoughtSignature: signature } });
        }
        // Reasoning blocks carry text this adapter never asks for; a Gemini thought
        // part needs its own signature to replay, so a foreign one is dropped.
    });
    return parts;
}
/** Project one user message onto Gemini parts. */
function userParts(message, names) {
    const parts = [];
    for (const block of message.content) {
        if (block.type === 'text') {
            if (typeof block.text === 'string' && block.text.length > 0)
                parts.push({ text: block.text });
            continue;
        }
        if (block.type === 'tool-result') {
            const callId = String(block.toolCallId);
            parts.push({
                functionResponse: {
                    // Vertex matches a response to its call by name, and by id when given.
                    name: names.get(callId) ?? callId,
                    ...callId.length > 0 ? { id: callId } : {},
                    response: { result: resultText((block.content ?? [])) },
                },
            });
        }
    }
    return parts;
}
/**
 * Build the request body for one model call.
 *
 * History is projected part by part — tool results become `functionResponse`
 * parts, replayed tool calls carry their thought signature — and consecutive
 * same-role turns are merged, which is the one turn shape Gemini accepts.
 *
 * No `thinkingConfig` is ever sent: Gemini's own default (dynamic thinking) is
 * what keeps 2.5 Pro working, since that model refuses a zero thinking budget.
 * @param options - the harness request.
 * @param config - project, location, and default output cap.
 * @returns the wire body.
 */
export function buildGeminiRequest(options, config) {
    const model = options.model;
    const names = toolNames(options.messages);
    const contents = [];
    for (const message of options.messages) {
        if (message.role === 'system')
            continue;
        const role = message.role === 'assistant' ? 'model' : 'user';
        const parts = role === 'model' ? assistantParts(message, model) : userParts(message, names);
        if (parts.length === 0)
            continue;
        const previous = contents.at(-1);
        if (previous !== undefined && previous.role === role)
            previous.parts.push(...parts);
        else
            contents.push({ role, parts });
    }
    const system = systemText(options);
    const tools = (options.tools ?? []).map(tool => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
    }));
    const generationConfig = {
        maxOutputTokens: options.maxTokens ?? config.maxTokens,
        ...options.temperature === undefined ? {} : { temperature: options.temperature },
        ...options.stop === undefined || options.stop.length === 0 ? {} : { stopSequences: [...options.stop] },
    };
    return {
        contents,
        ...system.length === 0 ? {} : { systemInstruction: { parts: [{ text: system }] } },
        ...tools.length === 0 ? {} : { tools: [{ functionDeclarations: tools }] },
        generationConfig,
    };
}
/**
 * Map Gemini's counters onto harness accounting.
 *
 * Gemini folds cached input into `promptTokenCount` and reports it separately as
 * well, so the harness's disjoint rule means subtracting it out; thinking tokens
 * are billed as output, exactly as the harness's own pi-ai mapping treats them.
 * @param usage - the latest cumulative counters seen on the stream.
 * @returns disjoint harness counts.
 */
export function mapGeminiUsage(usage) {
    const cached = usage.cachedContentTokenCount ?? 0;
    const prompt = usage.promptTokenCount;
    const thoughts = usage.thoughtsTokenCount ?? 0;
    const candidates = usage.candidatesTokenCount;
    const output = (candidates ?? 0) + thoughts;
    // An exact total is the provider's own; the arithmetic fallback is only as
    // complete as the two counters it needs, so a missing one omits the total
    // rather than reporting a partial sum as the whole request.
    const total = usage.totalTokenCount
        ?? (prompt === undefined || candidates === undefined ? undefined : prompt + output);
    return {
        inputTokens: Math.max(0, (prompt ?? 0) - cached),
        outputTokens: output,
        ...total === undefined ? {} : { totalTokens: total },
        ...cached > 0 ? { cacheReadTokens: cached } : {},
        ...thoughts > 0 ? { reasoningTokens: thoughts } : {},
    };
}
/**
 * Map Gemini's finish reason onto the harness vocabulary.
 *
 * `STOP`, `OTHER`, and an absent reason all mean the same thing here, and so
 * does any reason a provider release adds: the turn ended, and only a tool call
 * in it changes what that is called.
 * @param reason - the `finishReason` Vertex reported.
 * @param sawToolCall - whether the response contained a function call, which
 *   Gemini reports as an ordinary `STOP`.
 * @returns the harness finish reason.
 */
export function mapGeminiFinishReason(reason, sawToolCall) {
    switch (reason) {
        case 'MAX_TOKENS':
            return { kind: 'max-tokens' };
        case 'SAFETY':
        case 'RECITATION':
        case 'BLOCKLIST':
        case 'PROHIBITED_CONTENT':
        case 'SPII':
        case 'IMAGE_SAFETY':
            return {
                kind: 'error',
                failure: {
                    message: `google-vertex: the model refused to answer (finish reason ${reason})`,
                    code: SAFETY_BLOCKED_CODE,
                },
            };
        case 'MALFORMED_FUNCTION_CALL':
            return {
                kind: 'error',
                failure: {
                    message: 'google-vertex: the model produced a malformed function call',
                    code: 'INVALID_REQUEST',
                },
            };
        default:
            return sawToolCall ? { kind: 'tool-calls' } : { kind: 'stop' };
    }
}
/**
 * Translate Gemini's `streamGenerateContent` chunks into harness chunks.
 *
 * Gemini streams whole parts rather than deltas, so the shape of this translator
 * is: text parts append to one open text block, a function call closes that block
 * and is itself closed immediately (arguments arrive complete), and the terminal
 * chunk carries usage plus the stop reason.
 */
export class GeminiStreamTranslator {
    #model;
    #open;
    #nextIndex = 0;
    #replay = [];
    #usage = {};
    #usageReported = false;
    #finishReason;
    #sawFinish = false;
    #failed = false;
    #sawToolCall = false;
    /**
     * @param model - the model being called, recorded in the replay envelope.
     */
    constructor(model) {
        this.#model = model;
    }
    /**
     * Feed one decoded SSE payload.
     * @param event - the parsed chunk.
     * @returns the chunks this payload completes, in order.
     */
    handle(event) {
        if (this.#failed)
            return [];
        // Vertex can refuse mid-stream with an error envelope instead of a
        // candidate. Without this the body simply ends, and the adapter would
        // report a truncated response and lose the provider's own message and code.
        const error = event['error'];
        if (error !== undefined) {
            this.#failed = true;
            return [{ type: 'finish', reason: { kind: 'error', failure: failureForEvent(error) } }];
        }
        const out = [];
        const candidates = event['candidates'];
        if (Array.isArray(candidates)) {
            for (const candidate of candidates) {
                const fields = asObject(candidate);
                if (fields === undefined)
                    continue;
                const content = asObject(fields['content']);
                const parts = content?.['parts'];
                if (Array.isArray(parts))
                    for (const part of parts)
                        out.push(...this.#part(asObject(part)));
                const reason = fields['finishReason'];
                if (typeof reason === 'string') {
                    this.#finishReason = reason;
                    this.#sawFinish = true;
                }
            }
        }
        const usage = asObject(event['usageMetadata']);
        if (usage !== undefined)
            this.#mergeUsage(usage);
        return out;
    }
    /** Absorb one content part. */
    #part(part) {
        if (part === undefined)
            return [];
        // A thought summary is reasoning text this adapter never requested; keeping
        // it would need its own signature to replay, which the harness block cannot
        // carry, so it is dropped rather than made unreplayable.
        if (part['thought'] === true)
            return [];
        const signature = typeof part['thoughtSignature'] === 'string' ? part['thoughtSignature'] : undefined;
        const call = asObject(part['functionCall']);
        if (call !== undefined) {
            const out = this.#closeText();
            const index = this.#nextIndex;
            this.#nextIndex += 1;
            const name = typeof call['name'] === 'string' ? call['name'] : '';
            const id = typeof call['id'] === 'string' && call['id'].length > 0 ? call['id'] : `call_${index}`;
            const args = asObject(call['args']) ?? {};
            this.#replay.push({ type: 'tool-call', ...signature === undefined ? {} : { thoughtSignature: signature } });
            this.#sawToolCall = true;
            out.push({ type: 'block-start', index, blockType: 'tool-call' }, { type: 'tool-call-delta', index, id, name, argumentsDelta: JSON.stringify(args) }, { type: 'block-end', index, block: { type: 'tool-call', id, name, arguments: JSON.stringify(args) } });
            return out;
        }
        const text = part['text'];
        if (typeof text === 'string' && text.length > 0) {
            const open = this.#open;
            if (open !== undefined) {
                open.text += text;
                if (signature !== undefined)
                    this.#setSignature(signature);
                return [{ type: 'text-delta', index: open.index, text }];
            }
            const index = this.#nextIndex;
            this.#nextIndex += 1;
            this.#open = { index, kind: 'text', text, signature };
            this.#replay.push({ type: 'text', ...signature === undefined ? {} : { thoughtSignature: signature } });
            return [{ type: 'block-start', index, blockType: 'text' }, { type: 'text-delta', index, text }];
        }
        return [];
    }
    /** Replace the open block's recorded signature, if the provider sent one. */
    #setSignature(signature) {
        const open = this.#open;
        if (open === undefined)
            return;
        const index = this.#replay.length - 1;
        const entry = this.#replay[index];
        if (entry === undefined || entry.type !== open.kind)
            return;
        this.#replay[index] = { type: entry.type, thoughtSignature: signature };
        open.signature = signature;
    }
    /** Close an open text block, emitting its authoritative end. */
    #closeText() {
        const open = this.#open;
        if (open === undefined)
            return [];
        this.#open = undefined;
        return [{ type: 'block-end', index: open.index, block: { type: 'text', text: open.text } }];
    }
    /** Merge the newest cumulative usage counters. */
    #mergeUsage(usage) {
        const take = (name) => {
            const value = usage[name];
            return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
        };
        const prompt = take('promptTokenCount');
        const candidates = take('candidatesTokenCount');
        const cached = take('cachedContentTokenCount');
        const thoughts = take('thoughtsTokenCount');
        const total = take('totalTokenCount');
        const merged = {
            ...prompt === undefined ? {} : { promptTokenCount: prompt },
            ...candidates === undefined ? {} : { candidatesTokenCount: candidates },
            ...cached === undefined ? {} : { cachedContentTokenCount: cached },
            ...thoughts === undefined ? {} : { thoughtsTokenCount: thoughts },
            ...total === undefined ? {} : { totalTokenCount: total },
        };
        // Only a counter the provider actually sent counts as a report.
        if (Object.keys(merged).length === 0)
            return;
        this.#usageReported = true;
        this.#usage = { ...this.#usage, ...merged };
    }
    /** True once the provider reported a finish reason. */
    get sawFinish() {
        return this.#sawFinish;
    }
    /** True once an in-band error ended the stream, which `handle` already reported. */
    get failed() {
        return this.#failed;
    }
    /** {@inheritDoc StreamTranslatorLike.terminal} */
    get terminal() {
        return this.#failed;
    }
    /**
     * Terminal chunks: the closed tail, usage, and the finish reason.
     *
     * Called once, after the body ends. A response that produced nothing at all is
     * reported as an empty response rather than as a silent success.
     * @returns the trailing chunks in emission order.
     */
    finish() {
        const out = this.#closeText();
        const reason = mapGeminiFinishReason(this.#finishReason, this.#sawToolCall);
        if (this.#replay.length === 0 && reason.kind !== 'error') {
            return [{
                    type: 'finish',
                    reason: {
                        kind: 'error',
                        failure: {
                            message: 'google-vertex: the model completed the response with no content',
                            code: EMPTY_RESPONSE_CODE,
                        },
                    },
                }];
        }
        return [
            ...out,
            // Usage accompanies a real provider report; a synthesized zero would
            // claim a measurement that never happened.
            ...this.#usageReported ? [{ type: 'usage', usage: mapGeminiUsage(this.#usage) }] : [],
            { type: 'finish', reason, replayState: geminiReplayState(this.#model, this.#replay) },
        ];
    }
}
