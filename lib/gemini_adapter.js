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
import { attributionHeaders } from '@deepseek-ai/dsh-llm';
import { credentialFailure, errorBody, errorFinish, idleTimeoutFailure, transportFinish } from './adapter.js';
import { ServiceAccountTokens } from './auth.js';
import { buildGeminiRequest, DEFAULT_GEMINI_CONTEXT_WINDOW, DEFAULT_GEMINI_MAX_TOKENS, GeminiStreamTranslator, geminiEndpointFor, } from './gemini.js';
import { failureForStatus, streamSseRecords } from './wire.js';
/**
 * Duck-typed adapter over Vertex's Gemini publisher endpoint.
 *
 * `LlmRuntime` reaches adapters through plain method calls, so this object needs
 * no harness base class; the plugin's only runtime `@deepseek-ai/*` dependency
 * is `@deepseek-ai/dsh-llm`'s pure `attributionHeaders()` helper.
 */
export class GoogleVertexGeminiAdapter {
    #config;
    #tokens;
    #fetch;
    /**
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport and token-source overrides for tests.
     */
    constructor(config, options = {}) {
        this.#config = config;
        this.#fetch = options.fetch ?? ((input, init) => globalThis.fetch(input, init));
        this.#tokens = options.tokens ?? new ServiceAccountTokens(config.serviceAccount, { fetch: this.#fetch });
    }
    /** {@inheritDoc LlmAdapterLike.providerInfo} */
    providerInfo(provider) {
        return { id: provider, name: 'Google Vertex AI (Gemini)' };
    }
    /** No provider-owned retry policy; the harness defaults classify Vertex's own 429s. */
    providerRetryPolicy(_provider) {
        return undefined;
    }
    /** No route charges visual tokens: this adapter is text-only. */
    imageRequestPricing(_provider, _model) {
        return undefined;
    }
    /** The configured Gemini catalog, in configuration order. */
    listModels(provider) {
        return Promise.resolve(this.#config.models.map(model => this.#info(provider, model.id)));
    }
    /** {@inheritDoc LlmAdapterLike.resolveModel} */
    resolveModel(provider, model, _signal) {
        const known = this.#config.models.find(entry => entry.id === model);
        return Promise.resolve({
            ...this.#info(provider, model),
            context: { contextWindow: known?.contextWindow ?? DEFAULT_GEMINI_CONTEXT_WINDOW },
            defaultMaxTokens: known?.maxTokens ?? DEFAULT_GEMINI_MAX_TOKENS,
        });
    }
    /** {@inheritDoc LlmAdapterLike.prepareCall} */
    async prepareCall(provider, model, signal) {
        return {
            model: await this.resolveModel(provider, model, signal),
            stream: (options) => this.stream(options),
        };
    }
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
    async *stream(options) {
        const model = options.model.length > 0 ? options.model : this.#config.models[0]?.id ?? '';
        const known = this.#config.models.find(entry => entry.id === model);
        const consumer = new AbortController();
        const signal = options.signal === undefined
            ? consumer.signal
            : AbortSignal.any([options.signal, consumer.signal]);
        let idleTimedOut = false;
        let idleTimer;
        const armIdle = () => {
            if (idleTimer !== undefined)
                clearTimeout(idleTimer);
            idleTimer = setTimeout(() => {
                idleTimedOut = true;
                consumer.abort('google-vertex: stream idle timeout');
            }, this.#config.streamIdleTimeoutMs);
        };
        const clearIdle = () => {
            if (idleTimer !== undefined)
                clearTimeout(idleTimer);
            idleTimer = undefined;
        };
        let token;
        armIdle();
        try {
            // The token mint is a network call too: an unbounded one stalls the same
            // way a stalled body does.
            token = await this.#tokens.get(signal);
        }
        catch (error) {
            if (idleTimedOut) {
                yield errorFinish(idleTimeoutFailure(this.#config.streamIdleTimeoutMs));
                return;
            }
            if (options.signal?.aborted === true) {
                yield transportFinish(options, error);
                return;
            }
            yield errorFinish(credentialFailure(error));
            return;
        }
        finally {
            clearIdle();
        }
        let response;
        armIdle();
        try {
            response = await this.#fetch(geminiEndpointFor(this.#config.project, this.#config.location, model), {
                method: 'POST',
                headers: {
                    'content-type': 'application/json',
                    'accept': 'text/event-stream',
                    ...attributionHeaders(),
                    'authorization': `Bearer ${token}`,
                },
                body: JSON.stringify(buildGeminiRequest(options, {
                    project: this.#config.project,
                    location: this.#config.location,
                    maxTokens: known?.maxTokens ?? this.#config.maxTokens,
                })),
                signal,
            });
        }
        catch (error) {
            if (idleTimedOut) {
                yield errorFinish(idleTimeoutFailure(this.#config.streamIdleTimeoutMs));
                return;
            }
            yield transportFinish(options, error);
            return;
        }
        finally {
            clearIdle();
        }
        if (!response.ok) {
            yield errorFinish(failureForStatus(response.status, await errorBody(response), `model "${model}" in ${this.#config.location}`));
            return;
        }
        if (response.body === null) {
            yield errorFinish({ message: 'google-vertex: response carried no body', code: 'TRANSPORT' });
            return;
        }
        const translator = new GeminiStreamTranslator(model);
        try {
            const events = streamSseRecords(response.body, { armIdle, clearIdle });
            for await (const event of events)
                yield* translator.handle(event);
        }
        catch (error) {
            // An in-band error already ended the turn, so a fault while closing the
            // body cannot add a second terminal chunk.
            if (translator.failed)
                return;
            if (idleTimedOut) {
                yield errorFinish(idleTimeoutFailure(this.#config.streamIdleTimeoutMs));
                return;
            }
            yield transportFinish(options, error);
            return;
        }
        finally {
            clearIdle();
        }
        // Exactly one terminal chunk leaves this stream: an in-band provider error
        // and a watchdog expiry already yielded theirs.
        if (translator.failed)
            return;
        if (!translator.sawFinish) {
            if (idleTimedOut) {
                yield errorFinish(idleTimeoutFailure(this.#config.streamIdleTimeoutMs));
                return;
            }
            if (options.signal?.aborted === true) {
                yield transportFinish(options, options.signal?.reason);
                return;
            }
            yield errorFinish({
                message: `google-vertex: model "${model}" stream ended before a finish reason`,
                code: 'TRANSPORT',
            });
            return;
        }
        yield* translator.finish();
    }
    /** Display metadata for one model id, named from the catalog when known. */
    #info(provider, model) {
        const known = this.#config.models.find(entry => entry.id === model);
        return {
            provider,
            id: model,
            name: known?.name ?? model,
            description: `Google Gemini on Vertex AI (project ${this.#config.project}, ${this.#config.location}).`,
            inputModalities: ['text'],
        };
    }
}
