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
import { streamVertex, VertexPublisherAdapter, } from './adapter.js';
import { buildGeminiRequest, DEFAULT_GEMINI_CONTEXT_WINDOW, DEFAULT_GEMINI_MAX_TOKENS, GeminiStreamTranslator, geminiEndpointFor, } from './gemini.js';
/**
 * The capacities every Gemini model serves, wherever the id came from. A
 * catalog entry carries no capacities of its own until a model actually needs
 * different ones.
 */
const GEMINI_CAPACITY = {
    contextWindow: DEFAULT_GEMINI_CONTEXT_WINDOW,
    defaultMaxTokens: DEFAULT_GEMINI_MAX_TOKENS,
};
/**
 * Duck-typed adapter over Vertex's Gemini publisher endpoint.
 *
 * The metadata face and the streaming pipeline are the shared ones; this class
 * supplies the Gemini catalog, wording, endpoint, body, and translator.
 */
export class GoogleVertexGeminiAdapter extends VertexPublisherAdapter {
    /**
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport and token-source overrides for tests.
     */
    constructor(config, options = {}) {
        super({
            providerName: 'Google Vertex AI (Gemini)',
            catalog: config.models,
            capacityFor: () => GEMINI_CAPACITY,
            describe: (_name, row) => `Google Gemini on Vertex AI (project ${row.project}, ${row.location}).`,
        }, config, options);
    }
    /**
     * Stream one completion through `:streamGenerateContent?alt=sse`.
     *
     * Gemini has no terminal event: the body simply ends, and the finish reason
     * rides the last content chunk. A body that ends without one is therefore a
     * truncated response, which is what {@link GeminiStreamTranslator.sawFinish}
     * distinguishes — unless an in-band error or the idle watchdog already ended
     * the turn. The shared pump owns the watchdog, the token mint, and the SSE
     * loop; this route's finish is built at the end of the body.
     */
    async *stream(options) {
        yield* streamVertex(this.config, options, this.fetch, this.tokens, model => new GeminiStreamTranslator(model), {
            endpoint: (model, config) => geminiEndpointFor(config.project, config.location, model),
            body: (request, _model, config) => buildGeminiRequest(request, {
                project: config.project,
                location: config.location,
                maxTokens: config.maxTokens,
            }),
            truncatedMessage: model => `google-vertex: model "${model}" stream ended before a finish reason`,
        });
    }
}
