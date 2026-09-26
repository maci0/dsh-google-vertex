/**
 * dsh-google-vertex — use Google-hosted models from Vertex AI inside DeepSeek
 * Harness, authenticated with a service-account file.
 *
 * Two capabilities, one configuration row: an `ctx.llm` provider adapter for
 * Vertex's Anthropic publisher endpoint (Claude) and one for its Google
 * publisher endpoint (Gemini). Registering them makes both routes selectable in
 * the Web client's model picker, because `buildModelCatalog()` enumerates
 * `ctx.llm.listProviders()` and asks each adapter for `listModels()` /
 * `resolveModel()`.
 *
 * A third: the `google-vertex` settings namespace. The Gemini adapter discovers
 * its catalog at runtime behind a five-minute cache, and a browser half has no
 * other way to reach this process, so the namespace's one write — the Refresh
 * control on this plugin's row page under Plugins — is what drops that cache
 * on demand.
 *
 * The credential is the service-account JSON itself: a path in configuration,
 * or `GOOGLE_APPLICATION_CREDENTIALS` in the launch environment. Nothing is
 * copied into the harness credential store, and the file is read once at mount
 * so a typo fails loudly instead of on the first message.
 *
 * @module dsh-google-vertex
 */
import Schema from '@deepseek-ai/schemastery';
import { GoogleVertexAnthropicAdapter } from './adapter.js';
import { loadServiceAccount } from './auth.js';
import { ServiceAccountTokens } from './auth.js';
import { fetchGeminiModels } from './discovery.js';
import { DEFAULT_GEMINI_CONTEXT_WINDOW, DEFAULT_GEMINI_MAX_TOKENS, DEFAULT_GEMINI_MODELS } from './gemini.js';
import { GoogleVertexGeminiAdapter } from './gemini_adapter.js';
import { DEFAULT_LOCATION, DEFAULT_STREAM_IDLE_TIMEOUT_MS, MAX_TIMER_DELAY_MS } from './wire.js';
/** Plugin name as it appears in the loader. */
export const name = 'google-vertex';
/** The `ctx.llm` route serving Google-hosted Anthropic (Claude) models. */
export const PROVIDER = 'google-vertex-anthropic';
/** The `ctx.llm` route serving Gemini models. */
export const GEMINI_PROVIDER = 'google-vertex-gemini';
/**
 * Settings namespace the browser half's card edits — the join key between the
 * two halves. The card registers into `plugins.row.config` under this namespace,
 * and the settings tab pairs the two without knowing what the namespace means.
 */
export const GOOGLE_VERTEX_SETTINGS_NAMESPACE = 'google-vertex';
/** The one service this plugin needs mounted. */
export const inject = ['llm'];
/**
 * Claude models Vertex serves, in picker order. Ids are the provider's own
 * aliases, which track the newest dated release of each family and need no
 * edit when Vertex promotes one; the catalog is overridable from configuration
 * for a deployment that pins dated versions or uses a different listing.
 */
export const DEFAULT_MODELS = [
    { id: 'claude-opus-4-6', name: 'Claude Opus 4.6 (Vertex)' },
    { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6 (Vertex)' },
    { id: 'claude-opus-4-5', name: 'Claude Opus 4.5 (Vertex)' },
    { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5 (Vertex)' },
    { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5 (Vertex)' },
];
/** Context capacity reported for every Claude model; every listed family serves 200k. */
export const DEFAULT_CONTEXT_WINDOW = 200_000;
/**
 * Output cap used when a caller omits one. Well under the models' 64k ceiling:
 * an agent turn rarely needs more, and the harness enforces its own budget.
 */
export const DEFAULT_MAX_TOKENS = 32_000;
/**
 * Row schema: defaults live here, so a deployment only states what it changes.
 *
 * `serviceAccountFile`, `project`, `models`, and `geminiModels` carry no
 * `.default()`: the first two fall back to the environment, and an omitted
 * catalog materializes empty, which `resolveConfig` treats exactly like an
 * absent one and replaces with the built-in list.
 *
 * `revalidatedAt` is volatile — the only kind of field the settings document
 * accepts — and carries no default: absence means "never refreshed manually".
 */
export const Config = Schema.object({
    serviceAccountFile: Schema.string(),
    project: Schema.string(),
    location: Schema.string().default(DEFAULT_LOCATION),
    models: Schema.array(Schema.string()),
    geminiModels: Schema.array(Schema.string()),
    contextWindow: Schema.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
    maxTokens: Schema.number().step(1).min(1).default(DEFAULT_MAX_TOKENS),
    streamIdleTimeoutMs: Schema.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
    revalidatedAt: Schema.string().volatile(),
});
/**
 * Validate and normalize one configuration row.
 *
 * Invalid values throw rather than being silently defaulted: a typo'd path or
 * project would otherwise present as an opaque provider error mid-turn.
 * @param config - raw row configuration.
 * @param env - environment consulted for the credential and region defaults.
 * @returns the resolved adapter configuration.
 */
export function resolveConfig(config = {}, env = process.env) {
    // The exported schema is the one source of the numeric defaults and bounds.
    // The credential path, the project, and the region keep their environment
    // fallbacks, so they are read off the raw row instead.
    //
    // `revalidatedAt` is dropped first: a mounted row carries it as a live
    // reference the string schema refuses, and the host never reads the value.
    const { revalidatedAt: _revalidation, ...raw } = config;
    const filled = Config(raw);
    const serviceAccountFile = filled.serviceAccountFile ?? env['GOOGLE_APPLICATION_CREDENTIALS'];
    if (serviceAccountFile === undefined || serviceAccountFile.trim().length === 0) {
        throw new Error('google-vertex: no service account configured — set serviceAccountFile in this plugin\'s row,'
            + ' or export GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path');
    }
    const serviceAccount = loadServiceAccount(serviceAccountFile);
    const project = filled.project
        ?? env['GOOGLE_CLOUD_PROJECT']
        ?? env['GCLOUD_PROJECT']
        ?? serviceAccount.project_id;
    if (project === undefined || project.trim().length === 0) {
        throw new Error(`google-vertex: no project configured — set project in this plugin's row, or export`
            + ` GOOGLE_CLOUD_PROJECT, or use a service-account file that carries "project_id"`);
    }
    const location = config.location ?? env['GOOGLE_CLOUD_LOCATION'] ?? DEFAULT_LOCATION;
    if (location.trim().length === 0 || !/^[a-z0-9-]+$/.test(location)) {
        throw new Error(`google-vertex: location "${location}" is not a valid Vertex region`);
    }
    const ids = filled.models === undefined || filled.models.length === 0
        ? DEFAULT_MODELS
        : filled.models.map(id => ({ id, name: id }));
    for (const model of ids) {
        if (model.id.trim().length === 0)
            throw new Error('google-vertex: model ids must be non-empty strings');
    }
    const geminiIds = filled.geminiModels === undefined || filled.geminiModels.length === 0
        ? DEFAULT_GEMINI_MODELS.map(model => model.id)
        : filled.geminiModels;
    const geminiModels = geminiIds.map((id) => {
        if (id.trim().length === 0)
            throw new Error('google-vertex: geminiModels ids must be non-empty strings');
        // A configured id keeps the built-in entry's wording when it is one of them;
        // every model serves the same capacities, which live on the wire config.
        return DEFAULT_GEMINI_MODELS.find(model => model.id === id) ?? { id, name: id };
    });
    const streamIdleTimeoutMs = filled.streamIdleTimeoutMs;
    return {
        serviceAccountFile,
        anthropic: {
            serviceAccount,
            project,
            location,
            models: ids,
            contextWindow: filled.contextWindow,
            maxTokens: filled.maxTokens,
            streamIdleTimeoutMs,
        },
        gemini: {
            serviceAccount,
            project,
            location,
            models: geminiModels,
            maxTokens: DEFAULT_GEMINI_MAX_TOKENS,
            streamIdleTimeoutMs,
        },
    };
}
/**
 * Mount both adapters.
 * @param ctx - host context; `ctx.llm` must be mounted (`inject` guarantees it).
 * @param config - this plugin's row configuration.
 */
export function apply(ctx, config = {}) {
    const resolved = resolveConfig(config);
    const { anthropic, gemini } = resolved;
    // Only the Gemini route discovers: Vertex has no Anthropic listing endpoint,
    // so the Claude catalog is the configured (or built-in) list itself.
    const fetchFn = (input, init) => globalThis.fetch(input, init);
    const tokenSource = new ServiceAccountTokens(anthropic.serviceAccount, { fetch: fetchFn });
    const discoverGemini = () => fetchGeminiModels(gemini.location, tokenSource, fetchFn);
    const anthropicAdapter = new GoogleVertexAnthropicAdapter(anthropic);
    const geminiAdapter = new GoogleVertexGeminiAdapter(gemini, { discover: discoverGemini });
    ctx.llm.registerAdapter([PROVIDER], anthropicAdapter);
    ctx.llm.registerAdapter([GEMINI_PROVIDER], geminiAdapter);
    // A profile edit of `revalidatedAt` drops both cached catalogs.
    if (typeof ctx.on === 'function') {
        ctx.on('loader/volatile-update', () => {
            anthropicAdapter.invalidateModels();
            geminiAdapter.invalidateModels();
        });
    }
    ctx.logger.info(`google-vertex: providers "${PROVIDER}" and "${GEMINI_PROVIDER}" registered for project ${anthropic.project}`
        + ` (location ${anthropic.location}, credentials ${resolved.serviceAccountFile})`
        + ` — Claude: ${anthropic.models.map(model => model.id).join(', ')}`
        + ` — Gemini: ${gemini.models.map(model => model.id).join(', ')} (+ live discovery)`);
}
