/**
 * Provider adapter for Google's own Gemini models on Vertex AI.
 *
 * Same credential as the Claude route beside it — one service-account file, one
 * bearer token per request — over the `publishers/google` endpoint instead of
 * `publishers/anthropic`. Both live in one plugin row so project, region, and
 * credentials are configured once.
 *
 * The route is text-only: `inputModalities: ['text']` makes `LlmRuntime` project
 * images and files to placeholder text before dispatch. Tool calling is fully
 * supported, including Vertex's required thought-signature replay.
 *
 * @module dsh-google-vertex/gemini-adapter
 */
import { VertexPublisherAdapter } from './adapter.ts';
import type { TokenProvider } from './adapter.ts';
import type { FetchLike, ServiceAccount } from './auth.ts';
import { type GeminiModel, type GeminiWireConfig } from './gemini.ts';
import type { GenerateOptions, StreamChunk } from './host.ts';
/** Resolved adapter configuration for the Gemini route. */
export interface GeminiAdapterConfig extends GeminiWireConfig {
    readonly serviceAccount: ServiceAccount;
    readonly models: readonly GeminiModel[];
    /** Bound on the interval between two stream reads, in milliseconds. */
    readonly streamIdleTimeoutMs: number;
}
/**
 * Duck-typed adapter over Vertex's Gemini publisher endpoint.
 *
 * The metadata face and the streaming pipeline are the shared ones; this class
 * supplies the Gemini catalog, wording, endpoint, body, and translator.
 */
export declare class GoogleVertexGeminiAdapter extends VertexPublisherAdapter<GeminiAdapterConfig> {
    /**
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport and token-source overrides for tests.
     */
    constructor(config: GeminiAdapterConfig, options?: {
        fetch?: FetchLike;
        tokens?: TokenProvider;
    });
    /**
     * Stream one completion through `:streamGenerateContent?alt=sse`.
     *
     * Gemini has no terminal event: the body simply ends, and the finish reason
     * rides the last content chunk. A body that ends without one is therefore a
     * truncated response, which is what {@link GeminiStreamTranslator.sawFinish}
     * distinguishes — unless an in-band error or the idle watchdog already ended
     * the turn. The shared pump owns the watchdog, the token mint, and the SSE
     * loop; this route's finish is built at the end of the body.
     */
    stream(options: GenerateOptions): AsyncIterable<StreamChunk>;
}
