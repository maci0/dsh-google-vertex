/**
 * Wire translation for Google-hosted Anthropic models on Vertex AI: the request
 * body Vertex accepts, the two endpoint shapes (global and regional), the SSE
 * event stream it answers with, and the failure envelope it refuses with.
 *
 * Everything here is pure: the adapter feeds it bytes and yields the chunks it
 * returns, so the protocol is testable without a harness or a network.
 *
 * Vertex serves Claude through the publisher endpoint rather than Anthropic's
 * own API, which changes three things this module owns: the path
 * (`…/publishers/anthropic/models/{model}:streamRawPredict`), a body carrying
 * `anthropic_version` and no `model` field, and a bearer token instead of an
 * `x-api-key` header.
 *
 * @module dsh-google-vertex/wire
 */
import type { ContentBlock, FinishReason, GenerateOptions, LlmFailure, StreamChunk, TokenUsage } from './host.ts';
/** Endpoint host used when no region is configured. */
export declare const DEFAULT_LOCATION = "global";
/**
 * Default bound on the interval between two stream reads, in milliseconds.
 * Matches the shipped remote adapters (`llm-deepseek/src/common/defaults.ts`).
 */
export declare const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300000;
/** Largest delay `setTimeout` schedules without clamping it to one millisecond. */
export declare const MAX_TIMER_DELAY_MS = 2147483647;
/**
 * Host for a location. `global` has no region prefix — the documented global
 * endpoint is `aiplatform.googleapis.com`, and `global-aiplatform…` is not a
 * host Vertex answers on.
 * @param location - configured or defaulted region, or `global`.
 * @returns the origin, without a trailing slash.
 */
export declare function endpointOrigin(location: string): string;
/**
 * The streaming publisher path for one model.
 * @param project - Google Cloud project id.
 * @param location - region, or `global`.
 * @param model - publisher model id, e.g. `claude-sonnet-4-5`.
 * @returns the absolute request URL.
 */
export declare function endpointFor(project: string, location: string, model: string): string;
/** Cache marker Vertex honours on tools, system blocks, and message blocks. */
interface CacheControl {
    readonly type: 'ephemeral';
}
/** One text block on the wire. */
interface WireTextBlock {
    type: 'text';
    text: string;
    cache_control?: CacheControl;
}
/** One tool-result block on the wire. */
interface WireToolResultBlock {
    type: 'tool_result';
    tool_use_id: string;
    content: string;
    is_error?: boolean;
    cache_control?: CacheControl;
}
/** One tool-use block on the wire. */
interface WireToolUseBlock {
    type: 'tool_use';
    id: string;
    name: string;
    input: unknown;
    cache_control?: CacheControl;
}
/** Any wire content block this adapter emits. */
type WireContentBlock = WireTextBlock | WireToolResultBlock | WireToolUseBlock;
/** One wire message. */
export interface WireMessage {
    role: 'user' | 'assistant';
    content: WireContentBlock[];
}
/** One wire tool declaration. */
interface WireTool {
    name: string;
    description: string;
    input_schema: Record<string, unknown>;
    cache_control?: CacheControl;
}
/** The request body `:streamRawPredict` accepts. */
interface WireRequestBody {
    anthropic_version: string;
    stream: true;
    max_tokens: number;
    messages: WireMessage[];
    system?: WireTextBlock[];
    tools?: WireTool[];
    temperature?: number;
    stop_sequences?: string[];
}
/** What the adapter needs to build a request and address the endpoint. */
export interface VertexWireConfig {
    project: string;
    location: string;
    /** Output cap materialized when a caller omits `maxTokens`. */
    maxTokens: number;
}
/**
 * Flatten nested tool-result content to the text Vertex accepts.
 *
 * The Gemini route sends a tool result the same way, so this is exported rather
 * than written twice.
 */
export declare function resultText(blocks: readonly ContentBlock[]): string;
/**
 * Build the request body for one model call.
 *
 * History is projected block by block, consecutive same-role messages are
 * merged (the provider reads one turn per role), and three cache breakpoints are
 * placed: end of tools, end of system, and the final block of the conversation.
 * Those three make the static prefix and every earlier turn cacheable while the
 * growing tail is re-read.
 * @param options - the harness request.
 * @param config - project, location, and default output cap.
 * @returns the wire body.
 */
export declare function buildRequestBody(options: GenerateOptions, config: VertexWireConfig): WireRequestBody;
/** Raw usage counters as Vertex reports them. */
interface WireUsage {
    input_tokens?: number;
    output_tokens?: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
}
/**
 * Map Vertex's usage counters onto harness accounting.
 *
 * Vertex splits prompt tokens the way the harness does — `input_tokens` counts
 * only uncached input, with cache reads and writes reported separately — so the
 * fields transfer without arithmetic, and the total is their sum.
 * @param usage - the latest cumulative counters seen on the stream.
 * @returns disjoint harness counts.
 */
export declare function mapUsage(usage: WireUsage): TokenUsage;
/** Harness code reported when the request overflowed the model's context. */
export declare const CONTEXT_WINDOW_EXCEEDED_CODE = "CONTEXT_WINDOW_EXCEEDED";
/** Harness code reported when the request produced nothing at all. */
export declare const EMPTY_RESPONSE_CODE = "EMPTY_RESPONSE";
/** Harness code reported when a quota or rate cap refused the request. */
export declare const QUOTA_EXCEEDED_CODE = "QUOTA";
/**
 * Map one provider stop reason onto the harness vocabulary.
 *
 * `pause_turn` needs no case of its own: it cannot recur here, because this
 * adapter declares no server-executed tools, so the answer is complete — which
 * is what the default already reports for any reason the provider adds.
 * @param reason - the `stop_reason` Vertex reported, if any.
 * @returns the harness finish reason.
 */
export declare function mapStopReason(reason: string | undefined): FinishReason;
/**
 * Classify a refused HTTP response.
 *
 * Two refusals carry a code of their own because the harness treats them
 * differently: an oversized request must not be retried, and an exhausted quota
 * is a capacity problem rather than an invalid one.
 * @param status - HTTP status.
 * @param body - response body text.
 * @param subject - route description named in the failure.
 * @returns the failure to report.
 */
export declare function failureForStatus(status: number, body: string, subject: string): LlmFailure;
/**
 * Classify a stream-level `error` payload.
 *
 * Both publishers deliver one mid-stream: Anthropic's envelope names the
 * condition in `type`, and Google's in a canonical `status` plus a numeric
 * `code`. A provider error is what turns a mid-stream refusal into a terminal
 * failure instead of a truncated response.
 * @param error - the payload's `error` member.
 * @returns the failure to report.
 */
export declare function failureForEvent(error: unknown): LlmFailure;
/**
 * Decode one SSE record (everything up to a blank line) into its JSON payload.
 *
 * `event:` lines are ignored because every Vertex payload carries its own
 * `type`; `data:` lines are concatenated, which is what the SSE specification
 * requires for a multi-line payload.
 * @param record - one raw record, without its trailing blank line.
 * @returns the parsed payload, or undefined for a comment, ping, or malformed record.
 */
export declare function parseSseRecord(record: string): Record<string, unknown> | undefined;
/**
 * Reassembles SSE records from arbitrarily split transport chunks.
 *
 * Line endings are normalized as text arrives, so a `\r\n` split across two
 * chunks still frames one record.
 */
export declare class SseBuffer {
    #private;
    /**
     * Absorb one decoded chunk.
     * @param text - newly decoded text.
     * @returns every complete record it completes, in order.
     */
    push(text: string): string[];
    /**
     * Release whatever a completed body ended with.
     * @returns the trailing record, or undefined when the body ended on a boundary.
     */
    flush(): string | undefined;
}
/** Per-read watchdog hooks the shared pump calls around every outstanding read. */
interface SsePumpHooks {
    /** Arm the idle bound for the read about to start. */
    armIdle(): void;
    /** Clear the armed idle bound; called as soon as a read resolves, and on exit. */
    clearIdle(): void;
}
/**
 * Frame one streaming response body into its parsed SSE payloads.
 *
 * Both publisher routes answer with the same framing and differ only in what
 * the payload means, so the read loop, the decoder and record buffers, the
 * end-of-body flush, and the trailing `SseBuffer` record live here once. The
 * adapter supplies the idle watchdog and translates each payload.
 *
 * The idle bound covers one outstanding read: it is armed before every read,
 * cleared as soon as that read resolves — a consumer holding a yielded event is
 * not a stalled provider — and cleared again when the pump exits, whether the
 * body ended, the reader stopped early, or a read threw. A read that throws is
 * left for the adapter to classify.
 * @param body - the response body's byte stream.
 * @param hooks - the caller's idle-watchdog controls.
 * @yields every payload the body framed, in order.
 */
export declare function streamSseRecords(body: AsyncIterable<Uint8Array>, hooks: SsePumpHooks): AsyncGenerator<Record<string, unknown>>;
/**
 * Translate Vertex's Anthropic event stream into harness chunks.
 *
 * The translator is stateful because the two protocols disagree about
 * granularity: Vertex announces a content block and then streams deltas, while
 * the harness wants a start, the deltas, and an authoritative close. It is
 * tolerant of the events it does not project — `ping`, thinking, citations,
 * server tool use — because a provider that adds one must not break the turn.
 */
export declare class StreamTranslator {
    #private;
    /**
     * Feed one decoded event.
     * @param event - the parsed SSE payload.
     * @returns the chunks this event completes, in order.
     */
    handle(event: Record<string, unknown>): StreamChunk[];
    /** True once a terminal event arrived, so the adapter can tell truncation. */
    get done(): boolean;
    /** {@inheritDoc StreamTranslatorLike.terminal} */
    get terminal(): boolean;
    /** {@inheritDoc StreamTranslatorLike.sawFinish} */
    get sawFinish(): boolean;
}
export {};
