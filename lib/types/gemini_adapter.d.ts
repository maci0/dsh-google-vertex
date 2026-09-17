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
import type { TokenProvider } from './adapter.ts';
import type { FetchLike, ServiceAccount } from './auth.ts';
import { type GeminiModel, type GeminiWireConfig } from './gemini.ts';
import type { GenerateOptions, LlmAdapterLike, LlmModelInfo, LlmProviderInfo, LlmResolvedModelInfo, PreparedAdapterCall, StreamChunk } from './host.ts';
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
 * `LlmRuntime` reaches adapters through plain method calls, so this object needs
 * no harness base class; the plugin's only runtime `@deepseek-ai/*` dependency
 * is `@deepseek-ai/dsh-llm`'s pure `attributionHeaders()` helper.
 */
export declare class GoogleVertexGeminiAdapter implements LlmAdapterLike {
    #private;
    /**
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport and token-source overrides for tests.
     */
    constructor(config: GeminiAdapterConfig, options?: {
        fetch?: FetchLike;
        tokens?: TokenProvider;
    });
    /** {@inheritDoc LlmAdapterLike.providerInfo} */
    providerInfo(provider: string): LlmProviderInfo;
    /** No provider-owned retry policy; the harness defaults classify Vertex's own 429s. */
    providerRetryPolicy(_provider: string): undefined;
    /** No route charges visual tokens: this adapter is text-only. */
    imageRequestPricing(_provider: string, _model: string): undefined;
    /** The configured Gemini catalog, in configuration order. */
    listModels(provider: string): Promise<readonly LlmModelInfo[]>;
    /** {@inheritDoc LlmAdapterLike.resolveModel} */
    resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo>;
    /** {@inheritDoc LlmAdapterLike.prepareCall} */
    prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall>;
    /**
     * Stream one completion through `:streamGenerateContent?alt=sse`.
     *
     * Gemini has no terminal event: the body simply ends, and the finish reason
     * rides the last content chunk. A body that ends without one is therefore a
     * truncated response, which is what {@link GeminiStreamTranslator.sawFinish}
     * distinguishes — unless an in-band error or the idle watchdog already ended
     * the turn.
     *
     * Every read is bounded by `streamIdleTimeoutMs`: a provider that stops
     * sending is a terminal `TIMEOUT` rather than a turn that never ends. The
     * watchdog owns its own controller so the stalled read can be torn down; the
     * caller's signal is combined with it when present.
     */
    stream(options: GenerateOptions): AsyncIterable<StreamChunk>;
}
