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

import { attributionHeaders } from '@deepseek-ai/dsh-llm'
import { ServiceAccountTokens, VertexAuthError } from './auth.ts'
import type { FetchLike, ServiceAccount } from './auth.ts'
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

/** Resolved adapter configuration. */
export interface VertexAnthropicConfig {
  readonly serviceAccount: ServiceAccount
  readonly project: string
  readonly location: string
  readonly models: readonly VertexModel[]
  /** Maximum combined request and response context reported for every model. */
  readonly contextWindow: number
  /** Output cap applied when a caller omits `maxTokens`. */
  readonly maxTokens: number
  /** Bound on the interval between two stream reads, in milliseconds. */
  readonly streamIdleTimeoutMs: number
}

/** Terminal error finish carrying one failure. */
export function errorFinish(failure: LlmFailure): StreamChunk {
  return { type: 'finish', reason: { kind: 'error', failure } }
}

/** Terminal aborted finish carrying one failure. */
export function abortedFinish(failure: LlmFailure): StreamChunk {
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

/**
 * Duck-typed adapter over Vertex's Anthropic publisher endpoint.
 *
 * `LlmRuntime` reaches adapters through plain method calls, so this object
 * needs no harness base class; the plugin's only runtime `@deepseek-ai/*`
 * dependency is `@deepseek-ai/dsh-llm`'s pure `attributionHeaders()` helper.
 */
export class GoogleVertexAnthropicAdapter implements LlmAdapterLike {
  readonly #config: VertexAnthropicConfig
  readonly #tokens: TokenProvider
  readonly #fetch: FetchLike

  /**
   * @param config - the resolved configuration this adapter serves.
   * @param options - transport and token-source overrides for tests.
   */
  constructor(
    config: VertexAnthropicConfig,
    options: { fetch?: FetchLike; tokens?: TokenProvider } = {},
  ) {
    this.#config = config
    this.#fetch = options.fetch ?? ((input, init) => globalThis.fetch(input, init))
    this.#tokens = options.tokens ?? new ServiceAccountTokens(config.serviceAccount, { fetch: this.#fetch })
  }

  /** {@inheritDoc LlmAdapterLike.providerInfo} */
  providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Google Vertex AI (Anthropic)' }
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

  /** No route charges visual tokens: this adapter is text-only. */
  imageRequestPricing(_provider: string, _model: string): undefined {
    return undefined
  }

  /** The configured Claude catalog, in configuration order. */
  listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve(this.#config.models.map(model => this.#info(provider, model.id)))
  }

  /** {@inheritDoc LlmAdapterLike.resolveModel} */
  resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({
      ...this.#info(provider, model),
      context: { contextWindow: this.#config.contextWindow },
      defaultMaxTokens: this.#config.maxTokens,
    })
  }

  /** {@inheritDoc LlmAdapterLike.prepareCall} */
  async prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall> {
    return {
      model: await this.resolveModel(provider, model, signal),
      stream: (options: GenerateOptions) => this.stream(options),
    }
  }

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
  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const model = options.model.length > 0 ? options.model : this.#config.models[0]?.id ?? ''
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
      }, this.#config.streamIdleTimeoutMs)
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
      token = await this.#tokens.get(signal)
    } catch (error) {
      if (idleTimedOut) {
        yield errorFinish(idleTimeoutFailure(this.#config.streamIdleTimeoutMs))
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

    const headers: Record<string, string> = {
      'content-type': 'application/json',
      'accept': 'text/event-stream',
      ...attributionHeaders(),
      'authorization': `Bearer ${token}`,
    }

    let response: Response
    armIdle()
    try {
      response = await this.#fetch(endpointFor(this.#config.project, this.#config.location, model), {
        method: 'POST',
        headers,
        body: JSON.stringify(buildRequestBody(options, this.#config)),
        signal,
      })
    } catch (error) {
      if (idleTimedOut) {
        yield errorFinish(idleTimeoutFailure(this.#config.streamIdleTimeoutMs))
        return
      }
      yield transportFinish(options, error)
      return
    } finally {
      clearIdle()
    }

    if (!response.ok) {
      const subject = `model "${model}" in ${this.#config.location}`
      yield errorFinish(failureForStatus(response.status, await errorBody(response), subject))
      return
    }
    if (response.body === null) {
      yield errorFinish({ message: 'google-vertex: response carried no body', code: 'TRANSPORT' })
      return
    }

    const translator = new StreamTranslator()
    try {
      const events = streamSseRecords(
        response.body as unknown as AsyncIterable<Uint8Array>,
        { armIdle, clearIdle },
      )
      for await (const event of events) yield * translator.handle(event)
    } catch (error) {
      // The provider already ended the turn, so a fault while closing the body
      // cannot add a second terminal chunk.
      if (translator.done) return
      if (idleTimedOut) {
        yield errorFinish(idleTimeoutFailure(this.#config.streamIdleTimeoutMs))
        return
      }
      yield transportFinish(options, error)
      return
    } finally {
      clearIdle()
    }

    // Exactly one terminal chunk leaves this stream. A body that ended without
    // the provider's terminal event is a truncated response, unless the watchdog
    // or the caller ended the turn, which is the more specific account of the
    // same missing event.
    if (translator.done) return
    if (idleTimedOut) {
      yield errorFinish(idleTimeoutFailure(this.#config.streamIdleTimeoutMs))
      return
    }
    if (options.signal?.aborted === true) {
      yield transportFinish(options, options.signal?.reason)
      return
    }
    yield errorFinish({
      message: `google-vertex: model "${model}" stream ended before message_stop`,
      code: 'TRANSPORT',
    })
  }

  /** Display metadata for one model id, named from the catalog when known. */
  #info(provider: string, model: string): LlmModelInfo {
    const known = this.#config.models.find(entry => entry.id === model)
    return {
      provider,
      id: model,
      name: known?.name ?? model,
      description: `Google-hosted Anthropic model on Vertex AI (project ${this.#config.project}, ${this.#config.location}).`,
      inputModalities: ['text'],
    }
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
