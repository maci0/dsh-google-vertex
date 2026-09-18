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
import type { FetchLike } from './auth.ts';
import type { TokenProvider, VertexModel } from './adapter.ts';
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
 * Probe the hardcoded Anthropic model catalog against Vertex and return only
 * the models that are actually reachable.
 *
 * Probes run with bounded concurrency so a large catalog does not overwhelm
 * the endpoint.
 * @param candidates - the hardcoded model catalog to validate.
 * @param project - Google Cloud project id.
 * @param location - region, or `global`.
 * @param tokens - token source for bearer authentication.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns the subset of candidates that responded with anything other than 404.
 */
export declare function probeAnthropicModels(candidates: readonly VertexModel[], project: string, location: string, tokens: TokenProvider, fetchFn: FetchLike, signal?: AbortSignal): Promise<readonly VertexModel[]>;
/**
 * A cached, TTL-bounded model list that falls back to a static default when
 * the remote fetch fails.
 *
 * Both adapters use one of these: the Gemini adapter fetches from the catalog
 * API, and the Anthropic adapter uses the static default (since Vertex has no
 * Anthropic model listing endpoint).
 */
export declare class ModelCache<T> {
    #private;
    /**
     * @param fallback - static default returned when the fetch fails or is not
     *   attempted.
     * @param ttlMs - cache lifetime in milliseconds; defaults to 5 minutes.
     */
    constructor(fallback: readonly T[], ttlMs?: number);
    /**
     * Return the cached model list, or fetch a fresh one.
     *
     * When a fetch function is provided and the cache is stale, it is called to
     * produce a fresh list. On failure, the fallback is returned. Concurrent
     * callers share one in-flight fetch.
     * @param fetchFn - optional async function that returns a fresh model list.
     * @returns the model list, from cache, fetch, or fallback.
     */
    get(fetchFn?: () => Promise<readonly T[]>): Promise<readonly T[]>;
    /** Force the next `get` to re-fetch. */
    invalidate(): void;
}
