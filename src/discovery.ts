/**
 * Dynamic model discovery for Vertex AI publisher endpoints.
 *
 * Gemini models are fetched from the Vertex Model Garden catalog API
 * (`publishers/google/models`). Anthropic models have no listing endpoint on
 * Vertex, so they fall back to the hardcoded defaults with an optional probe
 * that tests whether each model id is actually reachable.
 *
 * Results are cached with a configurable TTL so the model picker does not
 * make a network call on every open.
 *
 * @module dsh-google-vertex/discovery
 */

import type { FetchLike } from './auth.ts'
import type { TokenProvider, VertexModel } from './adapter.ts'
import { endpointOrigin } from './wire.ts'

/** How long a cached model list stays valid, in milliseconds. */
export const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000

/** Vertex API version for the model list endpoint. */
const API_VERSION = 'v1'

/** Response shape from the Vertex publishers/google/models list endpoint. */
interface PublisherModelsResponse {
  models?: PublisherModelEntry[]
  nextPageToken?: string
}

/** One model entry from the Vertex catalog. */
interface PublisherModelEntry {
  /** Resource name, e.g. `publishers/google/models/gemini-2.5-pro`. */
  name?: string
  /** Display name. */
  displayName?: string
  /** Model description. */
  description?: string
  /**
   * Supported methods — only models that support `generateContent` or
   * `streamGenerateContent` are useful to this adapter.
   */
  supportedActions?: string[]
}

/** Cached model list with an expiry timestamp. */
interface CachedModels<T> {
  models: readonly T[]
  expiresAt: number
}

/**
 * Extract the model id from a Vertex resource name.
 *
 * A resource name looks like `publishers/google/models/gemini-2.5-pro` — the
 * model id is the last segment. A name without a `/` is returned as-is.
 * @param resourceName - the `name` field from the catalog entry.
 * @returns the bare model id.
 */
function modelIdFromResource(resourceName: string): string {
  const lastSlash = resourceName.lastIndexOf('/')
  return lastSlash >= 0 ? resourceName.slice(lastSlash + 1) : resourceName
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
 *
 * The catalog lists all published models — not access-filtered — so every
 * model returned here should be callable with the configured project's
 * credentials. The caller filters to generative models.
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
 * Fetch all Gemini models from the Vertex publishers/google/models endpoint.
 *
 * Follows pagination and filters to models that support content generation.
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
  let url = listModelsUrl(location, 'google')
  let pageToken: string | undefined

  // Paginate through the catalog. Cap at 10 pages to avoid infinite loops
  // from a misbehaving endpoint.
  for (let page = 0; page < 10; page++) {
    const pageUrl = pageToken !== undefined ? `${url}?pageToken=${encodeURIComponent(pageToken)}` : url
    const response = await fetchPage(pageUrl, bearer, fetchFn, signal)

    for (const entry of response.models ?? []) {
      if (entry.name === undefined) continue
      const id = modelIdFromResource(entry.name)
      if (id.length === 0) continue

      // Only include models that support content generation.
      // The catalog may include embedding, vision-only, or code models
      // that this adapter cannot drive.
      const actions = entry.supportedActions ?? []
      const generative = actions.length === 0
        || actions.includes('generateContent')
        || actions.includes('streamGenerateContent')

      if (!generative) continue

      const name = entry.displayName ?? id
      models.push({ id, name: `${name} (Vertex)` })
    }

    pageToken = response.nextPageToken
    if (pageToken === undefined || pageToken.length === 0) break
  }

  return models
}

/**
 * The `:rawPredict` endpoint for one Anthropic model on Vertex.
 *
 * This is the non-streaming sibling of `:streamRawPredict` the adapter uses
 * for generation. A POST with an intentionally minimal body will return 400
 * (model exists, bad request) or 404 (model not served in this region).
 * @param project - Google Cloud project id.
 * @param location - region, or `global`.
 * @param model - publisher model id, e.g. `claude-sonnet-4-5`.
 * @returns the absolute probe URL.
 */
function anthropicProbeUrl(project: string, location: string, model: string): string {
  const origin = endpointOrigin(location)
  return `${origin}/${API_VERSION}/projects/${encodeURIComponent(project)}`
    + `/locations/${encodeURIComponent(location)}`
    + `/publishers/anthropic/models/${encodeURIComponent(model)}:rawPredict`
}

/**
 * Probe one Anthropic model to check whether it is reachable on this project
 * and region.
 *
 * Sends a minimal POST to `:rawPredict` with an empty messages array. The
 * response status tells us:
 * - **400** — model exists (bad request because the body is intentionally
 *   minimal)
 * - **404** — model is not served in this location or does not exist
 * - **200** — model exists (unlikely with an empty body, but still valid)
 * - **401/403** — credential issue; treated as "unknown", kept in the list
 * - other — treated as "unknown", kept in the list
 *
 * Only a definitive 404 removes a model from the catalog.
 * @param model - model id to probe.
 * @param project - Google Cloud project id.
 * @param location - region, or `global`.
 * @param token - bearer token.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns true if the model is reachable (or status is ambiguous).
 */
async function probeAnthropicModel(
  model: string,
  project: string,
  location: string,
  token: string,
  fetchFn: FetchLike,
  signal?: AbortSignal,
): Promise<boolean> {
  try {
    const response = await fetchFn(anthropicProbeUrl(project, location, model), {
      method: 'POST',
      headers: {
        'authorization': `Bearer ${token}`,
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        anthropic_version: 'vertex-2023-10-16',
        max_tokens: 1,
        messages: [],
      }),
      ...signal === undefined ? {} : { signal },
    })
    // Consume the body to free the connection.
    await response.text().catch(() => {})
    // 404 = model not found. Everything else (400, 200, 429, 500) = exists.
    return response.status !== 404
  } catch {
    // Network error — keep the model in the list rather than silently dropping it.
    return true
  }
}

/**
 * Probe the hardcoded Anthropic model catalog against Vertex and return only
 * the models that are actually reachable.
 * @param candidates - the hardcoded model catalog to validate.
 * @param project - Google Cloud project id.
 * @param location - region, or `global`.
 * @param tokens - token source for bearer authentication.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns the subset of candidates that responded with anything other than 404.
 */
export async function probeAnthropicModels(
  candidates: readonly VertexModel[],
  project: string,
  location: string,
  tokens: TokenProvider,
  fetchFn: FetchLike,
  signal?: AbortSignal,
): Promise<readonly VertexModel[]> {
  if (candidates.length === 0) return candidates
  const bearer = await tokens.get(signal)
  // The catalog is a handful of ids; one probe per id, all at once.
  const probes = candidates.map(model =>
    probeAnthropicModel(model.id, project, location, bearer, fetchFn, signal))
  const results = await Promise.all(probes)
  return candidates.filter((_, index) => results[index])
}

/**
 * A cached, TTL-bounded model list that falls back to a static default when
 * the remote fetch fails.
 *
 * Both adapters use one of these: the Gemini adapter fetches from the catalog
 * API, and the Anthropic adapter uses the static default (since Vertex has no
 * Anthropic model listing endpoint).
 */
export class ModelCache<T> {
  readonly #fallback: readonly T[]
  #cached: CachedModels<T> | undefined
  #inflight: Promise<readonly T[]> | undefined

  /**
   * @param fallback - static default returned when the fetch fails or is not
   *   attempted.
   */
  constructor(fallback: readonly T[]) {
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
  async get(fetchFn?: () => Promise<readonly T[]>): Promise<readonly T[]> {
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
      // Empty result — use fallback rather than showing nothing.
      return this.#fallback
    }).catch(() => {
      // Network failure — serve fallback silently.
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
