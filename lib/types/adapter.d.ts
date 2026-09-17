/**
 * Provider adapter for Google-hosted Anthropic models on Vertex AI.
 *
 * Two things separate this from the pi-ai-backed `google-vertex` route the
 * harness already ships. Its credential is a service-account file, turned into
 * a bearer token per request by {@link ServiceAccountTokens} rather than typed
 * in as an API key. And its models are Claude, which Vertex serves through
 * `publishers/anthropic` — a path and body the Gemini protocol cannot express.
 *
 * The route is text-only: `inputModalities: ['text']` makes `LlmRuntime`
 * project images and files to placeholder text before dispatch, which is honest
 * about what this adapter sends.
 *
 * @module dsh-google-vertex/adapter
 */
import type { FetchLike, ServiceAccount } from './auth.ts';
import type { GenerateOptions, LlmAdapterLike, LlmFailure, LlmModelInfo, LlmProviderInfo, LlmResolvedModelInfo, PreparedAdapterCall, StreamChunk } from './host.ts';
/** One advertised model. */
export interface VertexModel {
    readonly id: string;
    readonly name: string;
}
/**
 * The token source the adapter asks for a bearer token before each request.
 * Structural so a test can answer with a fixed token; {@link ServiceAccountTokens}
 * is the production implementation.
 */
export interface TokenProvider {
    get(signal?: AbortSignal): Promise<string>;
}
/** Resolved adapter configuration. */
export interface VertexAnthropicConfig {
    readonly serviceAccount: ServiceAccount;
    readonly project: string;
    readonly location: string;
    readonly models: readonly VertexModel[];
    /** Maximum combined request and response context reported for every model. */
    readonly contextWindow: number;
    /** Output cap applied when a caller omits `maxTokens`. */
    readonly maxTokens: number;
    /** Bound on the interval between two stream reads, in milliseconds. */
    readonly streamIdleTimeoutMs: number;
}
/** Terminal error finish carrying one failure. */
export declare function errorFinish(failure: LlmFailure): StreamChunk;
/** Terminal aborted finish carrying one failure. */
export declare function abortedFinish(failure: LlmFailure): StreamChunk;
/**
 * The failure for a stream that produced no bytes within the configured bound.
 * @param idleTimeoutMs - the configured per-read bound.
 * @returns the terminal failure, coded `TIMEOUT`.
 */
export declare function idleTimeoutFailure(idleTimeoutMs: number): LlmFailure;
/** Read a refused response's body, tolerating a transport that ends early. */
export declare function errorBody(response: Response): Promise<string>;
/**
 * Duck-typed adapter over Vertex's Anthropic publisher endpoint.
 *
 * `LlmRuntime` reaches adapters through plain method calls, so this object
 * needs no harness base class; the plugin's only runtime `@deepseek-ai/*`
 * dependency is `@deepseek-ai/dsh-llm`'s pure `attributionHeaders()` helper.
 */
export declare class GoogleVertexAnthropicAdapter implements LlmAdapterLike {
    #private;
    /**
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport and token-source overrides for tests.
     */
    constructor(config: VertexAnthropicConfig, options?: {
        fetch?: FetchLike;
        tokens?: TokenProvider;
    });
    /** {@inheritDoc LlmAdapterLike.providerInfo} */
    providerInfo(provider: string): LlmProviderInfo;
    /**
     * No provider-owned retry policy: Vertex's own 429s carry a quota code the
     * harness already classifies, and a per-request backoff belongs to the
     * harness defaults.
     *
     * This and {@link imageRequestPricing} exist because `LlmRuntime` calls them
     * on every dispatch; a duck-typed adapter must supply them or the first
     * registration throws.
     */
    providerRetryPolicy(_provider: string): undefined;
    /** No route charges visual tokens: this adapter is text-only. */
    imageRequestPricing(_provider: string, _model: string): undefined;
    /** The configured Claude catalog, in configuration order. */
    listModels(provider: string): Promise<readonly LlmModelInfo[]>;
    /** {@inheritDoc LlmAdapterLike.resolveModel} */
    resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo>;
    /** {@inheritDoc LlmAdapterLike.prepareCall} */
    prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall>;
    /**
     * Stream one completion through `:streamRawPredict`.
     *
     * The bearer token is minted before the request, so a credentials failure is
     * reported as a terminal finish rather than as a thrown error escaping the
     * generator; `LlmRuntime` would normalize a throw the same way, but the
     * classification here (`AUTH` versus `TRANSPORT`) is the actionable part.
     *
     * Every read is bounded by `streamIdleTimeoutMs`: a provider that stops
     * sending is a terminal `TIMEOUT` rather than a turn that never ends. The
     * watchdog owns its own controller so the stalled read can be torn down; the
     * caller's signal is combined with it when present.
     */
    stream(options: GenerateOptions): AsyncIterable<StreamChunk>;
}
/** Classify a credential failure raised before the request was sent. */
export declare function credentialFailure(error: unknown): LlmFailure;
/** Classify a fetch or body-read failure, distinguishing cancellation. */
export declare function transportFinish(options: GenerateOptions, error: unknown): StreamChunk;
