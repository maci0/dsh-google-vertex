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
import type { VertexAnthropicConfig, VertexModel } from './adapter.ts';
import type { GeminiAdapterConfig } from './gemini_adapter.ts';
import type { HostContext } from './host.ts';
/** Plugin name as it appears in the loader. */
export declare const name = "google-vertex";
/** The `ctx.llm` route serving Google-hosted Anthropic (Claude) models. */
export declare const PROVIDER = "google-vertex-anthropic";
/** The `ctx.llm` route serving Gemini models. */
export declare const GEMINI_PROVIDER = "google-vertex-gemini";
/**
 * Settings namespace the browser half's card edits — the join key between the
 * two halves. The card registers into `plugins.row.config` under this namespace,
 * and the settings tab pairs the two without knowing what the namespace means.
 */
export declare const GOOGLE_VERTEX_SETTINGS_NAMESPACE = "google-vertex";
/** The one service this plugin needs mounted. */
export declare const inject: string[];
/**
 * Claude models Vertex serves, in picker order. Ids are the provider's own
 * aliases, which track the newest dated release of each family and need no
 * edit when Vertex promotes one; the catalog is overridable from configuration
 * for a deployment that pins dated versions or uses a different listing.
 */
export declare const DEFAULT_MODELS: readonly VertexModel[];
/** Context capacity reported for every Claude model; every listed family serves 200k. */
export declare const DEFAULT_CONTEXT_WINDOW = 200000;
/**
 * Output cap used when a caller omits one. Well under the models' 64k ceiling:
 * an agent turn rarely needs more, and the harness enforces its own budget.
 */
export declare const DEFAULT_MAX_TOKENS = 32000;
/**
 * Configuration accepted from this plugin's row in a profile patch.
 *
 * The exported schema is what Cordis validates the row against and fills
 * defaults from; `resolveConfig` then normalizes the validated values.
 */
export interface Config {
    /** Service-account JSON path, `~` allowed; defaults to `GOOGLE_APPLICATION_CREDENTIALS`. */
    readonly serviceAccountFile?: string;
    /**
     * Project id; defaults to `GOOGLE_CLOUD_PROJECT`, then `GCLOUD_PROJECT`, then
     * the credentials file's own `project_id`.
     */
    readonly project?: string;
    /** Region, or `global` (the default); defaults to `GOOGLE_CLOUD_LOCATION`. */
    readonly location?: string;
    /** Claude model ids to advertise, replacing the built-in catalog. */
    readonly models?: string[];
    /** Gemini model ids to advertise, replacing the built-in catalog. */
    readonly geminiModels?: string[];
    /**
     * Claude context window reported for every model; defaults to 200000. The
     * Gemini route reports {@link DEFAULT_GEMINI_CONTEXT_WINDOW} for every model
     * it serves, which is why this key has no Gemini counterpart.
     */
    readonly contextWindow?: number;
    /** Claude output cap applied when a caller omits one; defaults to 32000. */
    readonly maxTokens?: number;
    /**
     * Bound on the interval between two stream reads on either route; defaults to
     * 300000. A provider that stops sending is reported as `TIMEOUT` instead of
     * holding the turn open forever.
     */
    readonly streamIdleTimeoutMs?: number;
    /**
     * Stamp written by the Refresh control; absent until the first manual refresh.
     * A plain string in the row, and volatile in the schema: the settings document
     * accepts only volatile fields, so the write commits into the running config
     * and its `loader/volatile-update` drops both cached catalogs. The host never
     * reads the value — the write itself is the signal.
     */
    readonly revalidatedAt?: string;
}
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
export declare const Config: Schema<Schemastery.ObjectS<NoInfer<{
    serviceAccountFile: Schema<string, string, "plain">;
    project: Schema<string, string, "plain">;
    location: Schema<string, string, "defined">;
    models: Schema<string[], string[], "plain">;
    geminiModels: Schema<string[], string[], "plain">;
    contextWindow: Schema<number, number, "defined">;
    maxTokens: Schema<number, number, "defined">;
    streamIdleTimeoutMs: Schema<number, number, "defined">;
    revalidatedAt: Schema<string, string, "volatile">;
}>>, Schemastery.ObjectT<NoInfer<{
    serviceAccountFile: Schema<string, string, "plain">;
    project: Schema<string, string, "plain">;
    location: Schema<string, string, "defined">;
    models: Schema<string[], string[], "plain">;
    geminiModels: Schema<string[], string[], "plain">;
    contextWindow: Schema<number, number, "defined">;
    maxTokens: Schema<number, number, "defined">;
    streamIdleTimeoutMs: Schema<number, number, "defined">;
    revalidatedAt: Schema<string, string, "volatile">;
}>>, "plain">;
/** Validated configuration plus the file it was read from. */
interface ResolvedConfig {
    readonly anthropic: VertexAnthropicConfig;
    readonly gemini: GeminiAdapterConfig;
    /** Absolute path of the service-account file, for the mount log line. */
    readonly serviceAccountFile: string;
}
/**
 * Validate and normalize one configuration row.
 *
 * Invalid values throw rather than being silently defaulted: a typo'd path or
 * project would otherwise present as an opaque provider error mid-turn.
 * @param config - raw row configuration.
 * @param env - environment consulted for the credential and region defaults.
 * @returns the resolved adapter configuration.
 */
export declare function resolveConfig(config?: Config, env?: NodeJS.ProcessEnv): ResolvedConfig;
/**
 * Mount both adapters.
 * @param ctx - host context; `ctx.llm` must be mounted (`inject` guarantees it).
 * @param config - this plugin's row configuration.
 */
export declare function apply(ctx: HostContext, config?: Config): void;
export {};
