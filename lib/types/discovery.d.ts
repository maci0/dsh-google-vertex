/**
 * Dynamic model discovery for Vertex AI publisher endpoints.
 *
 * Gemini models are fetched from the Vertex Model Garden catalog API
 * (`publishers/google/models`). Anthropic models have no listing endpoint on
 * Vertex, so the adapter serves the catalog from configuration instead.
 *
 * Results are cached with a configurable TTL so the model picker does not
 * make a network call on every open.
 *
 * @module dsh-google-vertex/discovery
 */
import type { FetchLike } from './auth.ts';
import type { TokenProvider, VertexModel } from './adapter.ts';
/** How long a cached model list stays valid, in milliseconds. */
export declare const DEFAULT_CACHE_TTL_MS: number;
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
export declare function fetchGeminiModels(location: string, tokens: TokenProvider, fetchFn: FetchLike, signal?: AbortSignal): Promise<readonly VertexModel[]>;
/**
 * A cached, TTL-bounded model list that falls back to a static default when
 * the remote fetch fails.
 *
 * Only the Gemini adapter fetches: Vertex has no Anthropic model listing
 * endpoint, so that route serves its configured catalog without one and never
 * calls this.
 */
export declare class ModelCache {
    #private;
    /**
     * @param fallback - static default returned when the fetch fails or is not
     *   attempted.
     */
    constructor(fallback: readonly VertexModel[]);
    /**
     * Return the cached model list, or fetch a fresh one.
     *
     * When a fetch function is provided and the cache is stale, it is called to
     * produce a fresh list. On failure, the fallback is returned. Concurrent
     * callers share one in-flight fetch.
     * @param fetchFn - optional async function that returns a fresh model list.
     * @returns the model list, from cache, fetch, or fallback.
     */
    get(fetchFn?: () => Promise<readonly VertexModel[]>): Promise<readonly VertexModel[]>;
    /** Force the next `get` to re-fetch. */
    invalidate(): void;
}
