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
 * The credential is the service-account JSON itself: a path in configuration,
 * or `GOOGLE_APPLICATION_CREDENTIALS` in the launch environment. Nothing is
 * copied into the harness credential store, and the file is read once at mount
 * so a typo fails loudly instead of on the first message.
 *
 * @module dsh-google-vertex
 */

import Schema from '@deepseek-ai/schemastery'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import { GoogleVertexAnthropicAdapter } from './adapter.ts'
import type { VertexAnthropicConfig, VertexModel } from './adapter.ts'
import { loadServiceAccount } from './auth.ts'
import { DEFAULT_GEMINI_MODELS, type GeminiModel } from './gemini.ts'
import { GoogleVertexGeminiAdapter } from './gemini_adapter.ts'
import type { GeminiAdapterConfig } from './gemini_adapter.ts'
import { DEFAULT_LOCATION, DEFAULT_STREAM_IDLE_TIMEOUT_MS, MAX_TIMER_DELAY_MS } from './wire.ts'
import type { HostContext } from './host.ts'

/** Plugin name as it appears in the loader. */
export const name = 'google-vertex'

/** The `ctx.llm` route serving Google-hosted Anthropic (Claude) models. */
export const PROVIDER = 'google-vertex-anthropic'

/** The `ctx.llm` route serving Gemini models. */
export const GEMINI_PROVIDER = 'google-vertex-gemini'

/** The one service this plugin needs mounted. */
export const inject = ['llm']

/**
 * Claude models Vertex serves, in picker order. Ids are the provider's own
 * aliases, which track the newest dated release of each family and need no
 * edit when Vertex promotes one; the catalog is overridable from configuration
 * for a deployment that pins dated versions or uses a different listing.
 */
export const DEFAULT_MODELS: readonly VertexModel[] = [
  { id: 'claude-opus-4-6', name: 'Claude Opus 4.6 (Vertex)' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6 (Vertex)' },
  { id: 'claude-opus-4-5', name: 'Claude Opus 4.5 (Vertex)' },
  { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5 (Vertex)' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5 (Vertex)' },
]

/** Context capacity reported for every Claude model; every listed family serves 200k. */
export const DEFAULT_CONTEXT_WINDOW = 200_000

/**
 * Output cap used when a caller omits one. Well under the models' 64k ceiling:
 * an agent turn rarely needs more, and the harness enforces its own budget.
 */
export const DEFAULT_MAX_TOKENS = 32_000

/**
 * Configuration accepted from this plugin's row in a profile patch.
 *
 * The exported schema is what Cordis validates the row against and fills
 * defaults from; `resolveConfig` then normalizes the validated values.
 */
export interface Config {
  /** Service-account JSON path, `~` allowed; defaults to `GOOGLE_APPLICATION_CREDENTIALS`. */
  readonly serviceAccountFile?: string
  /**
   * Project id; defaults to `GOOGLE_CLOUD_PROJECT`, then `GCLOUD_PROJECT`, then
   * the credentials file's own `project_id`.
   */
  readonly project?: string
  /** Region, or `global` (the default); defaults to `GOOGLE_CLOUD_LOCATION`. */
  readonly location?: string
  /** Claude model ids to advertise, replacing the built-in catalog. */
  readonly models?: string[]
  /** Gemini model ids to advertise, replacing the built-in catalog. */
  readonly geminiModels?: string[]
  /** Claude context window reported for every model; defaults to 200000. */
  readonly contextWindow?: number
  /** Claude output cap applied when a caller omits one; defaults to 32000. */
  readonly maxTokens?: number
  /**
   * Bound on the interval between two stream reads on either route; defaults to
   * 300000. A provider that stops sending is reported as `TIMEOUT` instead of
   * holding the turn open forever.
   */
  readonly streamIdleTimeoutMs?: number
}

/**
 * Row schema: defaults live here, so a deployment only states what it changes.
 *
 * `serviceAccountFile`, `project`, `models`, and `geminiModels` carry no
 * `.default()`: the first two fall back to the environment, and an omitted
 * catalog materializes empty, which `resolveConfig` treats exactly like an
 * absent one and replaces with the built-in list.
 */
export const Config: Schema<Config> = Schema.object({
  serviceAccountFile: Schema.string(),
  project: Schema.string(),
  location: Schema.string().default(DEFAULT_LOCATION),
  models: Schema.array(Schema.string()),
  geminiModels: Schema.array(Schema.string()),
  contextWindow: Schema.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
  maxTokens: Schema.number().step(1).min(1).default(DEFAULT_MAX_TOKENS),
  streamIdleTimeoutMs: Schema.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
})

/** Validated configuration plus the file it was read from. */
export interface ResolvedConfig {
  readonly anthropic: VertexAnthropicConfig
  readonly gemini: GeminiAdapterConfig
  /** Absolute path of the service-account file, for the mount log line. */
  readonly serviceAccountFile: string
}

/** A positive integer, or a failure naming the field. */
function positiveInteger(value: number, field: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`google-vertex: ${field} must be a positive integer, got ${String(value)}`)
  }
  return value
}

/** Every environment name this plugin falls back to. */
const ENVIRONMENT_NAMES = [
  'GOOGLE_APPLICATION_CREDENTIALS',
  'GOOGLE_CLOUD_PROJECT',
  'GCLOUD_PROJECT',
  'GOOGLE_CLOUD_LOCATION',
] as const

/**
 * The falling-back values for the names above, read from the launcher's
 * environment snapshot when the composition has one and from the inherited
 * process environment otherwise.
 *
 * The launcher's snapshot records which layer supplied each value (process,
 * project `.env`, Harness-home `.env`), which a flattened `process.env` cannot;
 * `launchEnvironmentOf` itself degrades to a process-only snapshot, so this is
 * only about reaching a host that has no `get`.
 * @param ctx - the host context this plugin was mounted on.
 * @returns the names this plugin reads, as a plain environment object.
 */
function environmentOf(ctx: HostContext): NodeJS.ProcessEnv {
  if (typeof ctx.get !== 'function') return process.env
  const environment = launchEnvironmentOf(ctx as unknown as Parameters<typeof launchEnvironmentOf>[0])
  return Object.fromEntries(ENVIRONMENT_NAMES.map(name => [name, environment.get(name)?.value]))
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
export function resolveConfig(config: Config = {}, env: NodeJS.ProcessEnv = process.env): ResolvedConfig {
  const serviceAccountFile = config.serviceAccountFile ?? env['GOOGLE_APPLICATION_CREDENTIALS']
  if (serviceAccountFile === undefined || serviceAccountFile.trim().length === 0) {
    throw new Error(
      'google-vertex: no service account configured — set serviceAccountFile in this plugin\'s row,'
      + ' or export GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path',
    )
  }
  const serviceAccount = loadServiceAccount(serviceAccountFile)

  const project = config.project
    ?? env['GOOGLE_CLOUD_PROJECT']
    ?? env['GCLOUD_PROJECT']
    ?? serviceAccount.project_id
  if (project === undefined || project.trim().length === 0) {
    throw new Error(
      `google-vertex: no project configured — set project in this plugin's row, or export`
      + ` GOOGLE_CLOUD_PROJECT, or use a service-account file that carries "project_id"`,
    )
  }

  const location = config.location ?? env['GOOGLE_CLOUD_LOCATION'] ?? DEFAULT_LOCATION
  if (location.trim().length === 0 || !/^[a-z0-9-]+$/.test(location)) {
    throw new Error(`google-vertex: location "${location}" is not a valid Vertex region`)
  }

  const ids: readonly VertexModel[] = config.models === undefined || config.models.length === 0
    ? DEFAULT_MODELS
    : config.models.map(id => ({ id, name: id }))
  for (const model of ids) {
    if (model.id.trim().length === 0) throw new Error('google-vertex: model ids must be non-empty strings')
  }

  const geminiIds: readonly string[] = config.geminiModels === undefined || config.geminiModels.length === 0
    ? DEFAULT_GEMINI_MODELS.map(model => model.id)
    : config.geminiModels
  const geminiModels: readonly GeminiModel[] = geminiIds.map((id) => {
    if (id.trim().length === 0) throw new Error('google-vertex: geminiModels ids must be non-empty strings')
    // A configured id keeps the built-in entry's capacities when it is one of
    // them, so the tighter Vertex limits stay applied to the models they belong
    // to; an unfamiliar id gets the family defaults.
    return DEFAULT_GEMINI_MODELS.find(model => model.id === id)
      ?? { id, name: id, contextWindow: DEFAULT_GEMINI_MODELS[0]?.contextWindow ?? 1_048_576, maxTokens: DEFAULT_GEMINI_MODELS[0]?.maxTokens ?? 65_535 }
  })

  // A bound `setTimeout` would clamp to 1ms, and zero or a fraction of a
  // millisecond is a typo rather than a policy.
  const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
  if (!Number.isFinite(streamIdleTimeoutMs) || streamIdleTimeoutMs <= 0 || streamIdleTimeoutMs > MAX_TIMER_DELAY_MS) {
    throw new Error(
      `google-vertex: streamIdleTimeoutMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS},`
      + ` got ${String(config.streamIdleTimeoutMs)}`,
    )
  }

  return {
    serviceAccountFile,
    anthropic: {
      serviceAccount,
      project,
      location,
      models: ids,
      contextWindow: positiveInteger(config.contextWindow ?? DEFAULT_CONTEXT_WINDOW, 'contextWindow'),
      maxTokens: positiveInteger(config.maxTokens ?? DEFAULT_MAX_TOKENS, 'maxTokens'),
      streamIdleTimeoutMs,
    },
    gemini: {
      serviceAccount,
      project,
      location,
      models: geminiModels,
      maxTokens: DEFAULT_GEMINI_MODELS[0]?.maxTokens ?? 65_535,
      streamIdleTimeoutMs,
    },
  }
}

/**
 * Mount both adapters.
 * @param ctx - host context; `ctx.llm` must be mounted (`inject` guarantees it).
 * @param config - this plugin's row configuration.
 */
export function apply(ctx: HostContext, config: Config = {}): void {
  const resolved = resolveConfig(config, environmentOf(ctx))
  const { anthropic, gemini } = resolved

  ctx.llm.registerAdapter([PROVIDER], new GoogleVertexAnthropicAdapter(anthropic))
  ctx.llm.registerAdapter([GEMINI_PROVIDER], new GoogleVertexGeminiAdapter(gemini))

  ctx.logger.info(
    `google-vertex: providers "${PROVIDER}" and "${GEMINI_PROVIDER}" registered for project ${anthropic.project}`
    + ` (location ${anthropic.location}, credentials ${resolved.serviceAccountFile})`
    + ` — Claude: ${anthropic.models.map(model => model.id).join(', ')}`
    + ` — Gemini: ${gemini.models.map(model => model.id).join(', ')}`,
  )
}
