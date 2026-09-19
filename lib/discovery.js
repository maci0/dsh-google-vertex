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
import { endpointOrigin } from './wire.js';
/** How long a cached model list stays valid, in milliseconds. */
export const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000;
/** Vertex API version for the model list endpoint. */
const API_VERSION = 'v1';
/**
 * Extract the model id from a Vertex resource name.
 *
 * A resource name looks like `publishers/google/models/gemini-2.5-pro` — the
 * model id is the last segment. A name without a `/` is returned as-is.
 * @param resourceName - the `name` field from the catalog entry.
 * @returns the bare model id.
 */
function modelIdFromResource(resourceName) {
    const lastSlash = resourceName.lastIndexOf('/');
    return lastSlash >= 0 ? resourceName.slice(lastSlash + 1) : resourceName;
}
/**
 * Build the URL that lists publisher models for one publisher.
 * @param location - configured Vertex region, or `global`.
 * @param publisher - publisher name, e.g. `google`.
 * @returns the absolute list URL.
 */
function listModelsUrl(location, publisher) {
    const origin = endpointOrigin(location);
    return `${origin}/${API_VERSION}/publishers/${encodeURIComponent(publisher)}/models`;
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
async function fetchPage(url, token, fetchFn, signal) {
    const response = await fetchFn(url, {
        method: 'GET',
        headers: {
            'authorization': `Bearer ${token}`,
            'accept': 'application/json',
        },
        ...signal === undefined ? {} : { signal },
    });
    if (!response.ok) {
        throw new Error(`google-vertex: model list failed (HTTP ${response.status})`);
    }
    const body = await response.json();
    if (typeof body !== 'object' || body === null)
        return {};
    return body;
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
export async function fetchGeminiModels(location, tokens, fetchFn, signal) {
    const bearer = await tokens.get(signal);
    const models = [];
    let url = listModelsUrl(location, 'google');
    let pageToken;
    // Paginate through the catalog. Cap at 10 pages to avoid infinite loops
    // from a misbehaving endpoint.
    for (let page = 0; page < 10; page++) {
        const pageUrl = pageToken !== undefined ? `${url}?pageToken=${encodeURIComponent(pageToken)}` : url;
        const response = await fetchPage(pageUrl, bearer, fetchFn, signal);
        for (const entry of response.models ?? []) {
            if (entry.name === undefined)
                continue;
            const id = modelIdFromResource(entry.name);
            if (id.length === 0)
                continue;
            // Only include models that support content generation.
            // The catalog may include embedding, vision-only, or code models
            // that this adapter cannot drive.
            const actions = entry.supportedActions ?? [];
            const generative = actions.length === 0
                || actions.includes('generateContent')
                || actions.includes('streamGenerateContent');
            if (!generative)
                continue;
            const name = entry.displayName ?? id;
            models.push({ id, name: `${name} (Vertex)` });
        }
        pageToken = response.nextPageToken;
        if (pageToken === undefined || pageToken.length === 0)
            break;
    }
    return models;
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
    #fallback;
    #cached;
    #inflight;
    /**
     * @param fallback - static default returned when the fetch fails or is not
     *   attempted.
     */
    constructor(fallback) {
        this.#fallback = fallback;
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
    async get(fetchFn) {
        // Return cached if still valid.
        const cached = this.#cached;
        if (cached !== undefined && Date.now() < cached.expiresAt)
            return cached.models;
        // No fetch function means static-only.
        if (fetchFn === undefined)
            return this.#fallback;
        // Share one in-flight fetch.
        if (this.#inflight !== undefined)
            return this.#inflight;
        const operation = fetchFn().then((models) => {
            if (models.length > 0) {
                this.#cached = { models, expiresAt: Date.now() + DEFAULT_CACHE_TTL_MS };
                return models;
            }
            // Empty result — use fallback rather than showing nothing.
            return this.#fallback;
        }).catch(() => {
            // Network failure — serve fallback silently.
            return this.#cached?.models ?? this.#fallback;
        }).finally(() => {
            if (this.#inflight === operation)
                this.#inflight = undefined;
        });
        this.#inflight = operation;
        return operation;
    }
    /** Force the next `get` to re-fetch. */
    invalidate() {
        this.#cached = undefined;
    }
}
