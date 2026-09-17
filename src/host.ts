/**
 * The slice of the DeepSeek Harness host surface this plugin uses, declared
 * structurally.
 *
 * Like the other plugins under `~/dsh-plugins`, this package is installed
 * outside the harness checkout and cannot resolve `@deepseek-ai/*` from its own
 * directory, so it carries no runtime dependency on them. `LlmRuntime` reaches
 * adapters through plain method calls — there is no `instanceof LlmAdapter`
 * check anywhere in it — so a duck-typed adapter is a supported shape.
 *
 * Each declaration is narrowed to what this adapter reads or emits. Fields the
 * adapter never touches are deliberately absent: a mirrored field that nothing
 * reads is a field whose absence goes unnoticed. Widen a declaration when the
 * adapter starts using it, not before.
 *
 * @module dsh-google-vertex/host
 */

/** Disposer returned by every host registration. */
export type Disposable = () => void

/** Content block the harness may hand us in request history. */
export type ContentBlock =
  | { readonly type: 'text'; readonly text: string }
  | { readonly type: 'reasoning'; readonly text: string }
  | { readonly type: 'tool-call'; readonly id: string; readonly name: string; readonly arguments: string }
  | { readonly type: 'tool-result'; readonly toolCallId: string; readonly content: readonly ContentBlock[]; readonly isError?: boolean }
  | { readonly type: string; readonly [key: string]: unknown }

/** One message in a fully-assembled request. */
export interface Message {
  readonly id: string
  readonly role: 'system' | 'user' | 'assistant'
  readonly content: readonly ContentBlock[]
  /**
   * Producer attribution. Only `replayState` is read: an assistant message this
   * adapter produced carries the provider's per-block metadata (Vertex's
   * thought signatures), which the next request must echo back.
   */
  readonly source?: {
    readonly kind?: string
    readonly replayState?: unknown
  }
}

/** JSON-schema description of one tool, as the harness sends it. */
export interface ToolSchema {
  readonly name: string
  readonly description: string
  readonly parameters: Record<string, unknown>
}

/** A single model request, narrowed to the fields this adapter reads. */
export interface GenerateOptions {
  readonly provider?: string
  readonly model: string
  readonly messages: readonly Message[]
  readonly system?: string
  readonly tools?: readonly ToolSchema[]
  readonly temperature?: number
  readonly maxTokens?: number
  readonly stop?: readonly string[]
  readonly signal?: AbortSignal
}

/** Token accounting for one model call. */
export interface TokenUsage {
  inputTokens: number
  outputTokens: number
  totalTokens?: number
  cacheReadTokens?: number
  cacheWriteTokens?: number
  reasoningTokens?: number
}

/** Stable provider-neutral failure shape. */
export interface LlmFailure {
  readonly message: string
  readonly code: string
  readonly status?: number
}

/** Why a model response stopped. */
export type FinishReason =
  | { readonly kind: 'stop' }
  | { readonly kind: 'tool-calls' }
  | { readonly kind: 'max-tokens' }
  | { readonly kind: 'aborted'; readonly failure: LlmFailure }
  | { readonly kind: 'error'; readonly failure: LlmFailure }

/** Adapter-private lossless-JSON state carried by a terminal finish chunk. */
export interface ReplayEnvelope {
  /** Response-level metadata (ids, native stop reason). */
  readonly response: unknown
  /** Per-block metadata, one entry per emitted block in stream order. */
  readonly blocks?: readonly unknown[]
}

/** The chunk shapes this adapter emits. */
export type StreamChunk =
  | { readonly type: 'block-start'; readonly index: number; readonly blockType: string }
  | { readonly type: 'text-delta'; readonly index: number; readonly text: string }
  | { readonly type: 'tool-call-delta'; readonly index: number; readonly id: string; readonly name?: string; readonly argumentsDelta: string }
  | { readonly type: 'block-end'; readonly index: number; readonly block: ContentBlock }
  | { readonly type: 'usage'; readonly usage: TokenUsage }
  | { readonly type: 'finish'; readonly reason: FinishReason; readonly replayState?: ReplayEnvelope }

/** Display metadata for one adapter-owned provider route. */
export interface LlmProviderInfo {
  readonly id: string
  readonly name: string
}

/** One adapter-advertised model. */
export interface LlmModelInfo {
  readonly provider: string
  readonly id: string
  readonly name: string
  readonly description?: string
  readonly inputModalities?: readonly string[]
}

/** Exact-route model metadata resolved by its owning adapter. */
export interface LlmResolvedModelInfo extends LlmModelInfo {
  readonly context?: { readonly contextWindow: number }
  readonly defaultMaxTokens?: number
}

/** What `prepareCall` binds: exact model metadata plus one-generation dispatch. */
export interface PreparedAdapterCall {
  readonly model: LlmResolvedModelInfo
  readonly stream: (options: GenerateOptions) => AsyncIterable<StreamChunk>
}

/**
 * The adapter face `ctx.llm.registerAdapter()` consumes.
 *
 * The list is the full public surface of the harness `LlmAdapter` base class,
 * including the members that only have defaults there. `LlmRuntime` calls
 * `providerRetryPolicy` and `imageRequestPricing` on every dispatch, so a
 * duck-typed adapter that omits them throws on the first registration rather
 * than falling back to the base-class default.
 */
export interface LlmAdapterLike {
  providerInfo(provider: string): LlmProviderInfo
  providerRetryPolicy(provider: string): unknown
  imageRequestPricing(provider: string, model: string): unknown
  listModels(provider: string): Promise<readonly LlmModelInfo[]>
  resolveModel(provider: string, model: string, signal?: AbortSignal): Promise<LlmResolvedModelInfo>
  prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall>
  stream(options: GenerateOptions): AsyncIterable<StreamChunk>
}

/** The `ctx.llm` seam, narrowed to the one call this plugin makes. */
export interface LlmServiceLike {
  registerAdapter(providers: string[], adapter: LlmAdapterLike): Disposable
}

/** The host context slice this plugin touches. */
export interface HostContext {
  readonly llm: LlmServiceLike
  readonly logger: {
    info(message: unknown): void
    warn(message: unknown): void
  }
  /**
   * Cordis's service/context lookup. Present on every real host context; a
   * bare test host may omit it, in which case the plugin reads the inherited
   * `process.env` directly.
   */
  get?(name: string): unknown
}
