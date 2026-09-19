/**
 * The machinery both Vertex publisher adapters share, plus the Claude route's
 * own adapter: one metadata face the harness calls on every dispatch, and one
 * streaming pipeline carrying the per-read idle watchdog, the token mint, the
 * fetch, the refusal classification, the SSE loop, and the single-terminal-chunk
 * guarantee.
 *
 * Two things separate the Claude route from the pi-ai-backed `google-vertex`
 * route the harness already ships. Its credential is a service-account file,
 * turned into a bearer token per request by {@link ServiceAccountTokens} rather
 * than typed in as an API key. And its models are Claude, which Vertex serves
 * through `publishers/anthropic` — a path and body the Gemini protocol cannot
 * express.
 *
 * Both routes are text-only: `inputModalities: ['text']` makes `LlmRuntime`
 * project images and files to placeholder text before dispatch, which is honest
 * about what these adapters send.
 *
 * The two routes differ in exactly three things — how a request is addressed
 * and built, what the provider's payloads mean, and what a body that ends
 * without a terminal event should be called. Each is an injected callback;
 * everything else lives here once.
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
/**
 * Configuration every Vertex adapter row carries. The Claude route adds only
 * its context window; the model capacities of a Gemini route live on the
 * catalog entries, which is why `models` is typed loosely here.
 */
export interface VertexAdapterConfig {
    readonly serviceAccount: ServiceAccount;
    readonly project: string;
    readonly location: string;
    readonly models: readonly VertexModel[];
    /** Output cap applied when a caller omits `maxTokens`. */
    readonly maxTokens: number;
    /** Bound on the interval between two stream reads, in milliseconds. */
    readonly streamIdleTimeoutMs: number;
}
/** Resolved adapter configuration for the Claude route. */
export interface VertexAnthropicConfig extends VertexAdapterConfig {
    /** Maximum combined request and response context reported for every model. */
    readonly contextWindow: number;
}
/**
 * What separates one publisher route from the other, as one parameter set: the
 * display name, the catalogue wording, and the capacities `resolveModel`
 * reports for a model id.
 */
export interface AdapterMetadata<C extends VertexAdapterConfig> {
    /** Display name the model picker shows for this route. */
    readonly providerName: string;
    /** Catalogue this route advertises, in picker order. */
    readonly catalog: readonly VertexModel[];
    /** Context window and output cap reported for every model on this route. */
    readonly capacity: {
        contextWindow: number;
        defaultMaxTokens: number;
    };
    /** Model description reported for this route, project and region included. */
    readonly describe: (name: string, config: C) => string;
}
/**
 * The failure for a stream that produced no bytes within the configured bound.
 * @param idleTimeoutMs - the configured per-read bound.
 * @returns the terminal failure, coded `TIMEOUT`.
 */
export declare function idleTimeoutFailure(idleTimeoutMs: number): LlmFailure;
/** Read a refused response's body, tolerating a transport that ends early. */
export declare function errorBody(response: Response): Promise<string>;
/** Classify a credential failure raised before the request was sent. */
export declare function credentialFailure(error: unknown): LlmFailure;
/** Classify a fetch or body-read failure, distinguishing cancellation. */
export declare function transportFinish(options: GenerateOptions, error: unknown): StreamChunk;
/**
 * One provider translator, as the shared pump reads it.
 *
 * `handle` is the only method that can emit a terminal chunk — an Anthropic
 * `message_stop` and a Gemini in-band error both do. `terminal` then reports
 * that it, or the watchdog, already ended the turn, so the pump never adds a
 * second finish. A route whose terminal event is not the only way a body ends
 * — Gemini streams whole chunks and simply stops — narrows the truncation
 * question with `sawFinish`: did the provider itself report that the turn was
 * complete? A stream that ends without one is truncated. When absent, the pump
 * falls back to `terminal`.
 */
export interface StreamTranslatorLike {
    handle(event: Record<string, unknown>): Iterable<StreamChunk>;
    readonly terminal: boolean;
    readonly sawFinish?: boolean;
    /** Terminal chunks a provider that closes its body with data needs. */
    finish?(): Iterable<StreamChunk>;
}
/** Everything one streaming call varies between the two publisher routes. */
export interface StreamPumpOptions<C extends VertexAdapterConfig> {
    /** The request URL for the chosen model. */
    readonly endpoint: (model: string, config: C) => string;
    /** The request body for the chosen model. */
    readonly body: (options: GenerateOptions, config: C) => unknown;
    /** Failure named when the body ended without the provider's finish. */
    readonly truncatedMessage: (model: string) => string;
}
/**
 * One streaming completion, shared by both publisher routes.
 *
 * The bearer token is minted before the request, so a credentials failure is
 * reported as a terminal finish rather than as a thrown error escaping the
 * generator; `LlmRuntime` would normalize a throw the same way, but the
 * classification here (`AUTH` versus `TRANSPORT`) is the actionable part.
 *
 * Every read is bounded by `streamIdleTimeoutMs`: a provider that stops sending
 * is a terminal `TIMEOUT` rather than a turn that never ends. The watchdog owns
 * its own controller so the stalled read can be torn down; the caller's signal
 * is combined with it when present. The token mint is armed and cleared with
 * the same bound, because it is a network call too.
 *
 * Exactly one terminal chunk leaves this stream. A body that ended without the
 * provider's terminal event is a truncated response, unless the watchdog or the
 * caller ended the turn, which is the more specific account of the same missing
 * event.
 * @param config - the resolved configuration this adapter serves.
 * @param options - the harness request.
 * @param fetch - transport, already defaulted by the adapter.
 * @param tokens - token source, already defaulted by the adapter.
 * @param makeTranslator - builds this route's payload translator for one model.
 * @param pump - this route's endpoint, body, and terminal wording.
 * @yields every chunk the provider's stream completes.
 */
export declare function streamVertex(config: VertexAdapterConfig, options: GenerateOptions, fetch: FetchLike, tokens: TokenProvider, makeTranslator: (model: string) => StreamTranslatorLike, pump: StreamPumpOptions<VertexAdapterConfig>): AsyncGenerator<StreamChunk>;
/**
 * Duck-typed base for both Vertex publisher adapters.
 *
 * `LlmRuntime` reaches adapters through plain method calls, so these objects
 * need no harness base class; the plugin's only runtime `@deepseek-ai/*`
 * dependency is `@deepseek-ai/dsh-llm`'s pure `attributionHeaders()` helper.
 * The metadata face below is identical for both routes, so it is written once
 * and parameterized by {@link AdapterMetadata}.
 */
export declare abstract class VertexPublisherAdapter<C extends VertexAdapterConfig> implements LlmAdapterLike {
    #private;
    /**
     * @param metadata - display name, catalog, capacities, and description text.
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport, token-source, and discovery overrides.
     */
    constructor(metadata: AdapterMetadata<C>, config: C, options?: {
        fetch?: FetchLike;
        tokens?: TokenProvider;
        discover?: () => Promise<readonly VertexModel[]>;
    });
    /** The config this adapter serves, for the stream pipeline below. */
    protected get config(): C;
    /** The transport this adapter was built with. */
    protected get fetch(): FetchLike;
    /** The token source this adapter asks before each request. */
    protected get tokens(): TokenProvider;
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
    /** No route charges visual tokens: these adapters are text-only. */
    imageRequestPricing(_provider: string, _model: string): undefined;
    /**
     * The current model catalog, fetched from the provider when discovery is
     * configured, falling back to the static catalog on failure.
     *
     * The result is cached with a five-minute TTL so the model picker does
     * not make a network call on every open.
     */
    listModels(provider: string): Promise<readonly LlmModelInfo[]>;
    /**
     * Drop the cached catalog, so the next `listModels` re-discovers from the
     * provider. Both adapters answer the plugin's manual refresh with this.
     */
    invalidateModels(): void;
    /** {@inheritDoc LlmAdapterLike.resolveModel} */
    resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo>;
    /** {@inheritDoc LlmAdapterLike.prepareCall} */
    prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall>;
    /** Stream one completion through this route's publisher endpoint. */
    abstract stream(options: GenerateOptions): AsyncIterable<StreamChunk>;
}
/**
 * Duck-typed adapter over Vertex's Anthropic publisher endpoint.
 *
 * The route is text-only and declares no server-executed tools, and every
 * Claude family in the default catalog serves the same 200k context, so a
 * model's capacities are the configured pair rather than a per-model entry.
 */
export declare class GoogleVertexAnthropicAdapter extends VertexPublisherAdapter<VertexAnthropicConfig> {
    /**
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport, token-source, and discovery overrides for tests.
     */
    constructor(config: VertexAnthropicConfig, options?: {
        fetch?: FetchLike;
        tokens?: TokenProvider;
        discover?: () => Promise<readonly VertexModel[]>;
    });
    /**
     * Stream one completion through `:streamRawPredict`.
     *
     * The shared pump owns the watchdog, the token mint, the SSE loop, and the
     * single terminal chunk; `message_stop` closes the turn mid-body, so this
     * route's finish rides that event.
     */
    stream(options: GenerateOptions): AsyncIterable<StreamChunk>;
}
