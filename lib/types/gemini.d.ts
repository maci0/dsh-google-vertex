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
import type { FinishReason, GenerateOptions, Message, StreamChunk, TokenUsage } from './host.ts';
/** Harness code reported when the provider refused on safety grounds. */
export declare const SAFETY_BLOCKED_CODE = "SAFETY";
/**
 * Gemini context capacity: every current Gemini model serves roughly a million
 * tokens, and the catalog below narrows nothing.
 */
export declare const DEFAULT_GEMINI_CONTEXT_WINDOW = 1048576;
/**
 * Output cap applied when a caller omits one.
 *
 * Vertex's ceiling here is EXCLUSIVE — `maxOutputTokens: 65536` is refused with
 * "supported range is from 1 (inclusive) to 65536 (exclusive)" — so the highest
 * accepted value is one less.
 */
export declare const DEFAULT_GEMINI_MAX_TOKENS = 65535;
/** One advertised Gemini model with the capacities Vertex enforces. */
export interface GeminiModel {
    readonly id: string;
    readonly name: string;
    readonly contextWindow: number;
    readonly maxTokens: number;
}
/**
 * Gemini models the global endpoint serves, in picker order. Ids are the
 * provider's own aliases, so a promoted release needs no edit here.
 */
export declare const DEFAULT_GEMINI_MODELS: readonly GeminiModel[];
/**
 * The streaming publisher path for one Gemini model.
 * @param project - Google Cloud project id.
 * @param location - region, or `global`.
 * @param model - publisher model id, e.g. `gemini-3.5-flash`.
 * @returns the absolute request URL, with the SSE response encoding.
 */
export declare function geminiEndpointFor(project: string, location: string, model: string): string;
/** One function call the model requested. */
export interface GeminiFunctionCall {
    name: string;
    args?: Record<string, unknown>;
    id?: string;
}
/** One function result being sent back. */
export interface GeminiFunctionResponse {
    name: string;
    id?: string;
    response: Record<string, unknown>;
}
/** One content part on the wire. */
export interface GeminiPart {
    text?: string;
    thought?: boolean;
    thoughtSignature?: string;
    functionCall?: GeminiFunctionCall;
    functionResponse?: GeminiFunctionResponse;
}
/** One wire content turn. */
export interface GeminiContent {
    role: 'user' | 'model';
    parts: GeminiPart[];
}
/** One wire tool declaration. */
export interface GeminiToolDeclaration {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
}
/** The request body `:streamGenerateContent` accepts. */
export interface GeminiRequestBody {
    contents: GeminiContent[];
    systemInstruction?: {
        parts: {
            text: string;
        }[];
    };
    tools?: {
        functionDeclarations: GeminiToolDeclaration[];
    }[];
    generationConfig?: {
        temperature?: number;
        maxOutputTokens?: number;
        stopSequences?: string[];
    };
}
/** What the Gemini adapter needs to build a request and address the endpoint. */
export interface GeminiWireConfig {
    project: string;
    location: string;
    maxTokens: number;
}
/**
 * Index-aligned provider metadata for one emitted block.
 *
 * Vertex requires the thought signature of a function call to be echoed when the
 * call is replayed with its result; dropping it is a 400
 * ("Function call is missing a thought_signature"). Text parts can carry one
 * too, so the entry is kept for both block kinds.
 */
export interface GeminiReplayBlock {
    readonly type: 'text' | 'tool-call';
    readonly thoughtSignature?: string;
}
/** The envelope the harness stores on an assistant message and hands back. */
export interface GeminiReplayEnvelope {
    readonly response: {
        readonly kind: 'google-vertex-gemini';
        readonly version: 1;
        readonly model: string;
    };
    readonly blocks: readonly GeminiReplayBlock[];
}
/**
 * Build the replay envelope for one finished response.
 * @param model - the model that produced the response.
 * @param blocks - per-block metadata in emitted order.
 * @returns the envelope for the terminal finish chunk.
 */
export declare function geminiReplayState(model: string, blocks: readonly GeminiReplayBlock[]): GeminiReplayEnvelope;
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
export declare function readGeminiReplay(message: Message, model: string): readonly GeminiReplayBlock[] | undefined;
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
export declare function buildGeminiRequest(options: GenerateOptions, config: GeminiWireConfig): GeminiRequestBody;
/** Raw usage counters as Vertex reports them. */
export interface GeminiUsageMetadata {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    cachedContentTokenCount?: number;
    thoughtsTokenCount?: number;
    totalTokenCount?: number;
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
export declare function mapGeminiUsage(usage: GeminiUsageMetadata): TokenUsage;
/**
 * Map Gemini's finish reason onto the harness vocabulary.
 * @param reason - the `finishReason` Vertex reported.
 * @param sawToolCall - whether the response contained a function call, which
 *   Gemini reports as an ordinary `STOP`.
 * @returns the harness finish reason.
 */
export declare function mapGeminiFinishReason(reason: string | undefined, sawToolCall: boolean): FinishReason;
/**
 * Translate Gemini's `streamGenerateContent` chunks into harness chunks.
 *
 * Gemini streams whole parts rather than deltas, so the shape of this translator
 * is: text parts append to one open text block, a function call closes that block
 * and is itself closed immediately (arguments arrive complete), and the terminal
 * chunk carries usage plus the stop reason.
 */
export declare class GeminiStreamTranslator {
    #private;
    /**
     * @param model - the model being called, recorded in the replay envelope.
     */
    constructor(model: string);
    /**
     * Feed one decoded SSE payload.
     * @param event - the parsed chunk.
     * @returns the chunks this payload completes, in order.
     */
    handle(event: Record<string, unknown>): StreamChunk[];
    /** True once the provider reported a finish reason. */
    get sawFinish(): boolean;
    /** True once an in-band error ended the stream, which `handle` already reported. */
    get failed(): boolean;
    /**
     * Terminal chunks: the closed tail, usage, and the finish reason.
     *
     * Called once, after the body ends. A response that produced nothing at all is
     * reported as an empty response rather than as a silent success.
     * @returns the trailing chunks in emission order.
     */
    finish(): StreamChunk[];
}
