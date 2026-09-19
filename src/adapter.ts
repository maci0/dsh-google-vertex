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

import { attributionHeaders } from '@deepseek-ai/dsh-llm'
import { ServiceAccountTokens, VertexAuthError } from './auth.ts'
import type { FetchLike, ServiceAccount } from './auth.ts'
import { ModelCache } from './discovery.ts'
import type {
  GenerateOptions,
  LlmAdapterLike,
  LlmFailure,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  PreparedAdapterCall,
  StreamChunk,
} from './host.ts'
import {
  buildRequestBody,
  endpointFor,
  failureForStatus,
  streamSseRecords,
  StreamTranslator,
} from './wire.ts'

/** One advertised model. */
export interface VertexModel {
  readonly id: string
  readonly name: string
}

/**
 * The token source the adapter asks for a bearer token before each request.
 * Structural so a test can answer with a fixed token; {@link ServiceAccountTokens}
 * is the production implementation.
 */
export interface TokenProvider {
  get(signal?: AbortSignal): Promise<string>
}

/**
 * Configuration every Vertex adapter row carries. The Claude route adds only
 * its context window; the model capacities of a Gemini route live on the
 * catalog entries, which is why `models` is typed loosely here.
 */
export interface VertexAdapterConfig {
  readonly serviceAccount: ServiceAccount
  readonly project: string
  readonly location: string
  readonly models: readonly VertexModel[]
  /** Output cap applied when a caller omits `maxTokens`. */
  readonly maxTokens: number
  /** Bound on the interval between two stream reads, in milliseconds. */
  readonly streamIdleTimeoutMs: number
}

/** Resolved adapter configuration for the Claude route. */
export interface VertexAnthropicConfig extends VertexAdapterConfig {
  /** Maximum combined request and response context reported for every model. */
  readonly contextWindow: number
}

/**
 * What separates one publisher route from the other, as one parameter set: the
 * display name, the catalogue wording, and the capacities `resolveModel`
 * reports for a model id.
 */
export interface AdapterMetadata<C extends VertexAdapterConfig> {
  /** Display name the model picker shows for this route. */
  readonly providerName: string
  /** Catalogue this route advertises, in picker order. */
  readonly catalog: readonly VertexModel[]
  /** Context window and output cap reported for every model on this route. */
  readonly capacity: { contextWindow: number; defaultMaxTokens: number }
  /** Model description reported for this route, project and region included. */
  readonly describe: (name: string, config: C) => string
}

/** Terminal error finish carrying one failure. */
function errorFinish(failure: LlmFailure): StreamChunk {
  return { type: 'finish', reason: { kind: 'error', failure } }
}

/** Terminal aborted finish carrying one failure. */
function abortedFinish(failure: LlmFailure): StreamChunk {
  return { type: 'finish', reason: { kind: 'aborted', failure } }
}

/**
 * The failure for a stream that produced no bytes within the configured bound.
 * @param idleTimeoutMs - the configured per-read bound.
 * @returns the terminal failure, coded `TIMEOUT`.
 */
export function idleTimeoutFailure(idleTimeoutMs: number): LlmFailure {
  return {
    message: `google-vertex: no stream data for ${idleTimeoutMs}ms (streamIdleTimeoutMs)`,
    code: 'TIMEOUT',
  }
}

/** Read a refused response's body, tolerating a transport that ends early. */
export async function errorBody(response: Response): Promise<string> {
  try {
    return await response.text()
  } catch {
    return ''
  }
}

/** Classify a credential failure raised before the request was sent. */
export function credentialFailure(error: unknown): LlmFailure {
  if (error instanceof VertexAuthError) {
    return { message: error.message, code: error.code }
  }
  return {
    message: `google-vertex: ${error instanceof Error ? error.message : String(error)}`,
    code: 'AUTH',
  }
}

/** Classify a fetch or body-read failure, distinguishing cancellation. */
export function transportFinish(options: GenerateOptions, error: unknown): StreamChunk {
  if (options.signal?.aborted === true) {
    return abortedFinish({ message: 'google-vertex: request aborted', code: 'ABORTED' })
  }
  return errorFinish({
    message: `google-vertex: ${error instanceof Error ? error.message : String(error)}`,
    code: 'TRANSPORT',
  })
}

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
  handle(event: Record<string, unknown>): Iterable<StreamChunk>
  readonly terminal: boolean
  readonly sawFinish?: boolean
  /** Terminal chunks a provider that closes its body with data needs. */
  finish?(): Iterable<StreamChunk>
}

/** Everything one streaming call varies between the two publisher routes. */
export interface StreamPumpOptions<C extends VertexAdapterConfig> {
  /** The request URL for the chosen model. */
  readonly endpoint: (model: string, config: C) => string
  /** The request body for the chosen model. */
  readonly body: (options: GenerateOptions, config: C) => unknown
  /** Failure named when the body ended without the provider's finish. */
  readonly truncatedMessage: (model: string) => string
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
export async function* streamVertex(
  config: VertexAdapterConfig,
  options: GenerateOptions,
  fetch: FetchLike,
  tokens: TokenProvider,
  makeTranslator: (model: string) => StreamTranslatorLike,
  pump: StreamPumpOptions<VertexAdapterConfig>,
): AsyncGenerator<StreamChunk> {
  const model = options.model.length > 0 ? options.model : config.models[0]?.id ?? ''
  const consumer = new AbortController()
  const signal = options.signal === undefined
    ? consumer.signal
    : AbortSignal.any([options.signal, consumer.signal])
  let idleTimedOut = false
  let idleTimer: NodeJS.Timeout | undefined
  const armIdle = (): void => {
    if (idleTimer !== undefined) clearTimeout(idleTimer)
    idleTimer = setTimeout(() => {
      idleTimedOut = true
      consumer.abort('google-vertex: stream idle timeout')
    }, config.streamIdleTimeoutMs)
  }
  const clearIdle = (): void => {
    if (idleTimer !== undefined) clearTimeout(idleTimer)
    idleTimer = undefined
  }

  let token: string
  armIdle()
  try {
    // The token mint is a network call too: an unbounded one stalls the same
    // way a stalled body does.
    token = await tokens.get(signal)
  } catch (error) {
    if (idleTimedOut) {
      yield errorFinish(idleTimeoutFailure(config.streamIdleTimeoutMs))
      return
    }
    if (options.signal?.aborted === true) {
      yield transportFinish(options, error)
      return
    }
    yield errorFinish(credentialFailure(error))
    return
  } finally {
    clearIdle()
  }

  let response: Response
  armIdle()
  try {
    response = await fetch(pump.endpoint(model, config), {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'accept': 'text/event-stream',
        ...attributionHeaders(),
        'authorization': `Bearer ${token}`,
      },
      body: JSON.stringify(pump.body(options, config)),
      signal,
    })
  } catch (error) {
    if (idleTimedOut) {
      yield errorFinish(idleTimeoutFailure(config.streamIdleTimeoutMs))
      return
    }
    yield transportFinish(options, error)
    return
  } finally {
    clearIdle()
  }

  if (!response.ok) {
    yield errorFinish(failureForStatus(
      response.status,
      await errorBody(response),
      `model "${model}" in ${config.location}`,
    ))
    return
  }
  if (response.body === null) {
    yield errorFinish({ message: 'google-vertex: response carried no body', code: 'TRANSPORT' })
    return
  }

  const translator = makeTranslator(model)
  try {
    const events = streamSseRecords(
      response.body as unknown as AsyncIterable<Uint8Array>,
      { armIdle, clearIdle },
    )
    for await (const event of events) {
      yield * translator.handle(event)
      // The provider ended the turn mid-body — `message_stop`, or an in-band
      // error envelope. Its finish has been emitted, so neither the end of the
      // body nor a close fault may add a second one.
      if (translator.terminal) return
    }
  } catch (error) {
    // The provider already ended the turn, so a fault while closing the body
    // cannot add a second terminal chunk.
    if (translator.terminal) return
    if (idleTimedOut) {
      yield errorFinish(idleTimeoutFailure(config.streamIdleTimeoutMs))
      return
    }
    yield transportFinish(options, error)
    return
  } finally {
    clearIdle()
  }

  if (idleTimedOut) {
    yield errorFinish(idleTimeoutFailure(config.streamIdleTimeoutMs))
    return
  }
  if (options.signal?.aborted === true) {
    yield transportFinish(options, options.signal?.reason)
    return
  }
  // The body ended without the provider's own finish: a truncated response,
  // which is the more specific account of a turn the caller or the watchdog
  // already ended.
  if (!(translator.sawFinish ?? translator.terminal)) {
    yield errorFinish({ message: pump.truncatedMessage(model), code: 'TRANSPORT' })
    return
  }
  yield * translator.finish?.() ?? []
}

/**
 * Duck-typed base for both Vertex publisher adapters.
 *
 * `LlmRuntime` reaches adapters through plain method calls, so these objects
 * need no harness base class; the plugin's only runtime `@deepseek-ai/*`
 * dependency is `@deepseek-ai/dsh-llm`'s pure `attributionHeaders()` helper.
 * The metadata face below is identical for both routes, so it is written once
 * and parameterized by {@link AdapterMetadata}.
 */
export abstract class VertexPublisherAdapter<C extends VertexAdapterConfig> implements LlmAdapterLike {
  readonly #metadata: AdapterMetadata<C>
  readonly #config: C
  readonly #tokens: TokenProvider
  readonly #fetch: FetchLike
  readonly #modelCache: ModelCache<VertexModel>
  readonly #discover: (() => Promise<readonly VertexModel[]>) | undefined

  /**
   * @param metadata - display name, catalog, capacities, and description text.
   * @param config - the resolved configuration this adapter serves.
   * @param options - transport, token-source, and discovery overrides.
   */
  constructor(
    metadata: AdapterMetadata<C>,
    config: C,
    options: { fetch?: FetchLike; tokens?: TokenProvider; discover?: () => Promise<readonly VertexModel[]> } = {},
  ) {
    this.#metadata = metadata
    this.#config = config
    this.#fetch = options.fetch ?? ((input, init) => globalThis.fetch(input, init))
    this.#tokens = options.tokens ?? new ServiceAccountTokens(config.serviceAccount, { fetch: this.#fetch })
    this.#discover = options.discover
    this.#modelCache = new ModelCache(metadata.catalog)
  }

  /** The config this adapter serves, for the stream pipeline below. */
  protected get config(): C {
    return this.#config
  }

  /** The transport this adapter was built with. */
  protected get fetch(): FetchLike {
    return this.#fetch
  }

  /** The token source this adapter asks before each request. */
  protected get tokens(): TokenProvider {
    return this.#tokens
  }

  /** {@inheritDoc LlmAdapterLike.providerInfo} */
  providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: this.#metadata.providerName }
  }

  /**
   * No provider-owned retry policy: Vertex's own 429s carry a quota code the
   * harness already classifies, and a per-request backoff belongs to the
   * harness defaults.
   *
   * This and {@link imageRequestPricing} exist because `LlmRuntime` calls them
   * on every dispatch; a duck-typed adapter must supply them or the first
   * registration throws.
   */
  providerRetryPolicy(_provider: string): undefined {
    return undefined
  }

  /** No route charges visual tokens: these adapters are text-only. */
  imageRequestPricing(_provider: string, _model: string): undefined {
    return undefined
  }

  /**
   * The current model catalog, fetched from the provider when discovery is
   * configured, falling back to the static catalog on failure.
   *
   * The result is cached with a five-minute TTL so the model picker does
   * not make a network call on every open.
   */
  async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const models = await this.#modelCache.get(this.#discover)
    return models.map(model => this.#info(provider, model.id, model.name))
  }

  /**
   * Drop the cached catalog, so the next `listModels` re-discovers from the
   * provider. Both adapters answer the plugin's manual refresh with this.
   */
  invalidateModels(): void {
    this.#modelCache.invalidate()
  }

  /** {@inheritDoc LlmAdapterLike.resolveModel} */
  resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    const capacity = this.#metadata.capacity
    return Promise.resolve({
      ...this.#info(provider, model),
      context: { contextWindow: capacity.contextWindow },
      defaultMaxTokens: capacity.defaultMaxTokens,
    })
  }

  /** {@inheritDoc LlmAdapterLike.prepareCall} */
  async prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall> {
    return {
      model: await this.resolveModel(provider, model, signal),
      stream: (options: GenerateOptions) => this.stream(options),
    }
  }

  /** Stream one completion through this route's publisher endpoint. */
  abstract stream(options: GenerateOptions): AsyncIterable<StreamChunk>

  /** Display metadata for one model id, named from the catalog or the given name. */
  #info(provider: string, model: string, overrideName?: string): LlmModelInfo {
    const known = this.#metadata.catalog.find(entry => entry.id === model)
    const name = overrideName ?? known?.name ?? model
    return {
      provider,
      id: model,
      name,
      description: this.#metadata.describe(name, this.#config),
      inputModalities: ['text'],
    }
  }
}

/**
 * Duck-typed adapter over Vertex's Anthropic publisher endpoint.
 *
 * The route is text-only and declares no server-executed tools, and every
 * Claude family in the default catalog serves the same 200k context, so a
 * model's capacities are the configured pair rather than a per-model entry.
 */
export class GoogleVertexAnthropicAdapter extends VertexPublisherAdapter<VertexAnthropicConfig> {
  /**
   * @param config - the resolved configuration this adapter serves.
   * @param options - transport, token-source, and discovery overrides for tests.
   */
  constructor(
    config: VertexAnthropicConfig,
    options: { fetch?: FetchLike; tokens?: TokenProvider; discover?: () => Promise<readonly VertexModel[]> } = {},
  ) {
    super({
      providerName: 'Google Vertex AI (Anthropic)',
      catalog: config.models,
      capacity: { contextWindow: config.contextWindow, defaultMaxTokens: config.maxTokens },
      describe: (_name, row) =>
        `Google-hosted Anthropic model on Vertex AI (project ${row.project}, ${row.location}).`,
    }, config, options)
  }

  /**
   * Stream one completion through `:streamRawPredict`.
   *
   * The shared pump owns the watchdog, the token mint, the SSE loop, and the
   * single terminal chunk; `message_stop` closes the turn mid-body, so this
   * route's finish rides that event.
   */
  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    yield * streamVertex(this.config, options, this.fetch, this.tokens, () => new StreamTranslator(), {
      endpoint: (model, config) => endpointFor(config.project, config.location, model),
      body: (request, config) => buildRequestBody(request, config),
      truncatedMessage: model => `google-vertex: model "${model}" stream ended before message_stop`,
    })
  }
}
