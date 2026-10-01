/**
 * Dynamic model discovery for Vertex AI publisher endpoints.
 *
 * Gemini models are fetched from the Vertex Model Garden catalog API
 * (`publishers/google/models`). Anthropic models have no listing endpoint on
 * Vertex, so the adapter serves the catalog from configuration instead.
 *
 * Results are cached for five minutes so the model picker does not
 * make a network call on every open.
 *
 * @module dsh-google-vertex/discovery
 */

import type { FetchLike } from './auth.ts'
import type { TokenProvider, VertexModel } from './adapter.ts'
import { DEFAULT_GEMINI_MODELS } from './gemini.ts'
import { endpointOrigin } from './wire.ts'

/** How long a cached model list stays valid, in milliseconds. */
export const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000

/**
 * Vertex API version for the model list endpoint. `publishers.models.list`
 * exists only in v1beta1; v1 serves `get` alone.
 */
const API_VERSION = 'v1beta1'

/** Most catalog pages read before giving up on a misbehaving endpoint. */
const MAX_PAGES = 10

/**
 * Id segments naming a Gemini variant this text route cannot drive: embedding,
 * speech, image output, and the Live API.
 */
const NON_TEXT_SEGMENTS: ReadonlySet<string> = new Set(['embedding', 'tts', 'image', 'live', 'audio'])

/** `ListPublisherModelsResponse`, narrowed to the fields read here. */
interface PublisherModelsResponse {
  publisherModels?: PublisherModelEntry[]
  nextPageToken?: string
}

/**
 * One `PublisherModel`, narrowed to the field read here. It carries no display
 * name, and its `supportedActions` describes console actions rather than API
 * verbs, so neither can name or filter a model.
 */
interface PublisherModelEntry {
  /** Resource name, e.g. `publishers/google/models/gemini-2.5-pro`. */
  name?: string
}

/** Cached model list with an expiry timestamp. */
interface CachedModels {
  models: readonly VertexModel[]
  expiresAt: number
}

/**
 * Extract the model id from a Vertex resource name.
 *
 * A resource name looks like `publishers/google/models/gemini-2.5-pro`: the
 * model id is the last segment. A name without a `/` is returned as-is.
 * @param resourceName - the `name` field from the catalog entry.
 * @returns the bare model id.
 */
function modelIdFromResource(resourceName: string): string {
  const lastSlash = resourceName.lastIndexOf('/')
  return lastSlash >= 0 ? resourceName.slice(lastSlash + 1) : resourceName
}

/**
 * True for a Gemini id this route can serve: the publisher catalog also lists
 * Imagen, Veo, embedding, and speech models.
 * @param id - bare model id.
 */
function isGeminiTextModel(id: string): boolean {
  return id.startsWith('gemini-') && !id.split('-').some(segment => NON_TEXT_SEGMENTS.has(segment))
}

/**
 * Build the URL that lists publisher models for one publisher.
 * @param location - configured Vertex region, or `global`.
 * @param publisher - publisher name, e.g. `google`.
 * @returns the absolute list URL.
 */
function listModelsUrl(location: string, publisher: string): string {
  const origin = endpointOrigin(location)
  return `${origin}/${API_VERSION}/publishers/${encodeURIComponent(publisher)}/models`
}

/**
 * Fetch one page of publisher models from the Vertex Model Garden catalog.
 * @param url - the list endpoint URL, possibly with a pageToken.
 * @param token - bearer token for authentication.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns the parsed response.
 */
async function fetchPage(
  url: string,
  token: string,
  fetchFn: FetchLike,
  signal?: AbortSignal,
): Promise<PublisherModelsResponse> {
  const response = await fetchFn(url, {
    method: 'GET',
    headers: {
      'authorization': `Bearer ${token}`,
      'accept': 'application/json',
    },
    ...signal === undefined ? {} : { signal },
  })
  if (!response.ok) {
    throw new Error(`google-vertex: model list failed (HTTP ${response.status})`)
  }
  const body: unknown = await response.json()
  if (typeof body !== 'object' || body === null) return {}
  return body as PublisherModelsResponse
}

/**
 * Fetch the Gemini text models from Vertex's `publishers/google/models` list.
 *
 * Follows pagination and keeps `gemini-` ids minus the variants named in
 * {@link NON_TEXT_SEGMENTS}. A built-in id keeps its built-in name; any other
 * is named by its id.
 * @param location - configured Vertex region, or `global`.
 * @param tokens - token source for bearer authentication.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns model ids and display names, in catalog order.
 */
export async function fetchGeminiModels(
  location: string,
  tokens: TokenProvider,
  fetchFn: FetchLike,
  signal?: AbortSignal,
): Promise<readonly VertexModel[]> {
  const bearer = await tokens.get(signal)
  const models: VertexModel[] = []
  const url = listModelsUrl(location, 'google')
  let pageToken: string | undefined

  for (let page = 0; page < MAX_PAGES; page++) {
    const pageUrl = pageToken !== undefined ? `${url}?pageToken=${encodeURIComponent(pageToken)}` : url
    const response = await fetchPage(pageUrl, bearer, fetchFn, signal)

    for (const entry of response.publisherModels ?? []) {
      if (typeof entry.name !== 'string') continue
      const id = modelIdFromResource(entry.name)
      if (!isGeminiTextModel(id)) continue
      models.push(DEFAULT_GEMINI_MODELS.find(model => model.id === id) ?? { id, name: `${id} (Vertex)` })
    }

    pageToken = response.nextPageToken
    if (pageToken === undefined || pageToken.length === 0) break
  }

  return models
}

/**
 * A cached, TTL-bounded model list that falls back to a static default when
 * the remote fetch fails.
 *
 * Only the Gemini adapter fetches: Vertex has no Anthropic model listing
 * endpoint, so that route serves its configured catalog without one and never
 * calls this.
 */
export class ModelCache {
  readonly #fallback: readonly VertexModel[]
  #cached: CachedModels | undefined
  #inflight: Promise<readonly VertexModel[]> | undefined

  /**
   * @param fallback - static default returned when the fetch fails or is not
   *   attempted.
   */
  constructor(fallback: readonly VertexModel[]) {
    this.#fallback = fallback
  }

  /**
   * Return the cached model list, or fetch a fresh one.
   *
   * When a fetch function is provided and the cache is stale, it is called to
   * produce a fresh list. On failure, the fallback is returned. Concurrent
   * callers share one in-flight fetch.
   * @param fetchFn - optional async function that returns a fresh model list.
   * @returns the model list, from cache, fetch, or fallback.
   */
  async get(fetchFn?: () => Promise<readonly VertexModel[]>): Promise<readonly VertexModel[]> {
    // Return cached if still valid.
    const cached = this.#cached
    if (cached !== undefined && Date.now() < cached.expiresAt) return cached.models

    // No fetch function means static-only.
    if (fetchFn === undefined) return this.#fallback

    // Share one in-flight fetch.
    if (this.#inflight !== undefined) return this.#inflight

    const operation = fetchFn().then((models) => {
      if (models.length > 0) {
        this.#cached = { models, expiresAt: Date.now() + DEFAULT_CACHE_TTL_MS }
        return models
      }
      // Empty result: use fallback rather than showing nothing.
      return this.#fallback
    }).catch(() => {
      // Network failure: serve fallback silently.
      return this.#cached?.models ?? this.#fallback
    }).finally(() => {
      if (this.#inflight === operation) this.#inflight = undefined
    })
    this.#inflight = operation
    return operation
  }

  /** Force the next `get` to re-fetch. */
  invalidate(): void {
    this.#cached = undefined
  }
}
