/**
 * Wire translation for Google-hosted Anthropic models on Vertex AI: the request
 * body Vertex accepts, the two endpoint shapes (global and regional), the SSE
 * event stream it answers with, and the failure envelope it refuses with.
 *
 * Everything here is pure: the adapter feeds it bytes and yields the chunks it
 * returns, so the protocol is testable without a harness or a network.
 *
 * Vertex serves Claude through the publisher endpoint rather than Anthropic's
 * own API, which changes three things this module owns: the path
 * (`…/publishers/anthropic/models/{model}:streamRawPredict`), a body carrying
 * `anthropic_version` and no `model` field, and a bearer token instead of an
 * `x-api-key` header.
 *
 * @module dsh-google-vertex/wire
 */

import type {
  ContentBlock,
  FinishReason,
  GenerateOptions,
  LlmFailure,
  StreamChunk,
  TokenUsage,
  ToolSchema,
} from './host.ts'

/** Vertex's Anthropic API version marker; required in every request body. */
export const VERTEX_ANTHROPIC_VERSION = 'vertex-2023-10-16'

/** Endpoint host used when no region is configured. */
export const DEFAULT_LOCATION = 'global'

/**
 * Default bound on the interval between two stream reads, in milliseconds.
 * Matches the shipped remote adapters (`llm-deepseek/src/common/defaults.ts`).
 */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000

/** Largest delay `setTimeout` schedules without clamping it to one millisecond. */
export const MAX_TIMER_DELAY_MS = 2_147_483_647

/**
 * Host for a location. `global` has no region prefix — the documented global
 * endpoint is `aiplatform.googleapis.com`, and `global-aiplatform…` is not a
 * host Vertex answers on.
 * @param location - configured or defaulted region, or `global`.
 * @returns the origin, without a trailing slash.
 */
export function endpointOrigin(location: string): string {
  return location === DEFAULT_LOCATION
    ? 'https://aiplatform.googleapis.com'
    : `https://${location}-aiplatform.googleapis.com`
}

/**
 * The streaming publisher path for one model.
 * @param project - Google Cloud project id.
 * @param location - region, or `global`.
 * @param model - publisher model id, e.g. `claude-sonnet-4-5`.
 * @returns the absolute request URL.
 */
export function endpointFor(project: string, location: string, model: string): string {
  const path = `/v1/projects/${encodeURIComponent(project)}/locations/${encodeURIComponent(location)}`
    + `/publishers/anthropic/models/${encodeURIComponent(model)}:streamRawPredict`
  return `${endpointOrigin(location)}${path}`
}

/** Cache marker Vertex honours on tools, system blocks, and message blocks. */
export interface CacheControl {
  readonly type: 'ephemeral'
}

/** One text block on the wire. */
export interface WireTextBlock {
  type: 'text'
  text: string
  cache_control?: CacheControl
}

/** One tool-result block on the wire. */
export interface WireToolResultBlock {
  type: 'tool_result'
  tool_use_id: string
  content: string
  is_error?: boolean
  cache_control?: CacheControl
}

/** One tool-use block on the wire. */
export interface WireToolUseBlock {
  type: 'tool_use'
  id: string
  name: string
  input: unknown
  cache_control?: CacheControl
}

/** Any wire content block this adapter emits. */
export type WireContentBlock = WireTextBlock | WireToolResultBlock | WireToolUseBlock

/** One wire message. */
export interface WireMessage {
  role: 'user' | 'assistant'
  content: WireContentBlock[]
}

/** One wire tool declaration. */
export interface WireTool {
  name: string
  description: string
  input_schema: Record<string, unknown>
  cache_control?: CacheControl
}

/** The request body `:streamRawPredict` accepts. */
export interface WireRequestBody {
  anthropic_version: string
  stream: true
  max_tokens: number
  messages: WireMessage[]
  system?: WireTextBlock[]
  tools?: WireTool[]
  temperature?: number
  stop_sequences?: string[]
}

/** What the adapter needs to build a request and address the endpoint. */
export interface VertexWireConfig {
  project: string
  location: string
  /** Output cap materialized when a caller omits `maxTokens`. */
  maxTokens: number
}

/**
 * One cache breakpoint is only useful where Vertex can store a prefix, which is
 * every block type this adapter emits — the request never carries the provider's
 * own non-cacheable blocks.
 */
const CACHE_CONTROL: CacheControl = { type: 'ephemeral' }

/** Flatten nested tool-result content to the text Vertex accepts. */
function resultText(blocks: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block.type === 'text' && typeof block.text === 'string') parts.push(block.text)
    else if (block.type === 'tool-call') parts.push(`[tool call] ${String(block.name)}(${String(block.arguments)})`)
    else if (block.type === 'tool-result') parts.push(resultText((block.content ?? []) as readonly ContentBlock[]))
  }
  const joined = parts.join('')
  // An empty tool result is still a result: the provider rejects empty content.
  return joined.length > 0 ? joined : '(no output)'
}

/** Parse a model-produced arguments string into the object Vertex requires. */
function toolInput(argumentsJson: string): unknown {
  try {
    const parsed: unknown = JSON.parse(argumentsJson)
    return typeof parsed === 'object' && parsed !== null ? parsed : {}
  } catch {
    // The harness assembles these strings from the model's own deltas; an
    // unparseable one is a provider-side truncation, and an empty object keeps
    // the turn revisable instead of failing the whole request.
    return {}
  }
}

/** Ensure a tool's JSON Schema is one Anthropic accepts. */
function inputSchema(parameters: Record<string, unknown>): Record<string, unknown> {
  return {
    type: 'object',
    ...parameters,
    properties: typeof parameters['properties'] === 'object' && parameters['properties'] !== null
      ? parameters['properties']
      : {},
  }
}

/** System-role text this request carries, in assembly order. */
function systemText(options: GenerateOptions): string[] {
  const parts: string[] = []
  if (options.system !== undefined && options.system.length > 0) parts.push(options.system)
  for (const message of options.messages) {
    if (message.role !== 'system') continue
    for (const block of message.content) {
      if (block.type === 'text' && typeof block.text === 'string' && block.text.length > 0) parts.push(block.text)
    }
  }
  return parts
}

/** Project one non-system harness message onto its wire content. */
function messageContent(message: GenerateOptions['messages'][number]): WireContentBlock[] {
  const content: WireContentBlock[] = []
  for (const block of message.content) {
    switch (block.type) {
      case 'text':
        if (typeof block.text === 'string' && block.text.length > 0) content.push({ type: 'text', text: block.text })
        break
      case 'tool-call':
        content.push({
          type: 'tool_use',
          id: String(block.id),
          name: String(block.name),
          input: toolInput(String(block.arguments)),
        })
        break
      case 'tool-result':
        content.push({
          type: 'tool_result',
          tool_use_id: String(block.toolCallId),
          content: resultText((block.content ?? []) as readonly ContentBlock[]),
          ...block.isError === true ? { is_error: true } : {},
        })
        break
      default:
        // Reasoning blocks carry no reusable text: Vertex requires a signed
        // thinking block to replay one, and this adapter never enables
        // extended thinking, so a stray block is history from another route.
        break
    }
  }
  return content
}

/**
 * Build the request body for one model call.
 *
 * History is projected block by block, consecutive same-role messages are
 * merged (the provider reads one turn per role), and three cache breakpoints are
 * placed: end of tools, end of system, and the final block of the conversation.
 * Those three make the static prefix and every earlier turn cacheable while the
 * growing tail is re-read.
 * @param options - the harness request.
 * @param config - project, location, and default output cap.
 * @returns the wire body.
 */
export function buildRequestBody(options: GenerateOptions, config: VertexWireConfig): WireRequestBody {
  const messages: WireMessage[] = []
  for (const message of options.messages) {
    if (message.role === 'system') continue
    const role = message.role === 'assistant' ? 'assistant' : 'user'
    const content = messageContent(message)
    if (content.length === 0) continue
    const previous = messages.at(-1)
    if (previous !== undefined && previous.role === role) previous.content.push(...content)
    else messages.push({ role, content })
  }

  const systemParts = systemText(options)
  const system: WireTextBlock[] = systemParts.map(text => ({ type: 'text', text }))
  const tools: WireTool[] = (options.tools ?? []).map((tool: ToolSchema) => ({
    name: tool.name,
    description: tool.description,
    input_schema: inputSchema(tool.parameters),
  }))

  const lastTool = tools.at(-1)
  if (lastTool !== undefined) lastTool.cache_control = CACHE_CONTROL
  const lastSystem = system.at(-1)
  if (lastSystem !== undefined) lastSystem.cache_control = CACHE_CONTROL
  const lastMessage = messages.at(-1)
  const lastBlock = lastMessage?.content.at(-1)
  if (lastBlock !== undefined) lastBlock.cache_control = CACHE_CONTROL

  return {
    anthropic_version: VERTEX_ANTHROPIC_VERSION,
    stream: true,
    max_tokens: options.maxTokens ?? config.maxTokens,
    messages,
    ...system.length === 0 ? {} : { system },
    ...tools.length === 0 ? {} : { tools },
    ...options.temperature === undefined ? {} : { temperature: options.temperature },
    ...options.stop === undefined || options.stop.length === 0 ? {} : { stop_sequences: [...options.stop] },
  }
}

/** Raw usage counters as Vertex reports them. */
export interface WireUsage {
  input_tokens?: number
  output_tokens?: number
  cache_read_input_tokens?: number
  cache_creation_input_tokens?: number
}

/**
 * Map Vertex's usage counters onto harness accounting.
 *
 * Vertex splits prompt tokens the way the harness does — `input_tokens` counts
 * only uncached input, with cache reads and writes reported separately — so the
 * fields transfer without arithmetic, and the total is their sum.
 * @param usage - the latest cumulative counters seen on the stream.
 * @returns disjoint harness counts.
 */
export function mapUsage(usage: WireUsage): TokenUsage {
  const input = usage.input_tokens ?? 0
  const output = usage.output_tokens ?? 0
  const cacheRead = usage.cache_read_input_tokens ?? 0
  const cacheWrite = usage.cache_creation_input_tokens ?? 0
  return {
    inputTokens: input,
    outputTokens: output,
    totalTokens: input + output + cacheRead + cacheWrite,
    ...cacheRead > 0 ? { cacheReadTokens: cacheRead } : {},
    ...cacheWrite > 0 ? { cacheWriteTokens: cacheWrite } : {},
  }
}

/** Harness code reported when the request overflowed the model's context. */
export const CONTEXT_WINDOW_EXCEEDED_CODE = 'CONTEXT_WINDOW_EXCEEDED'
/** Harness code reported when the request produced nothing at all. */
export const EMPTY_RESPONSE_CODE = 'EMPTY_RESPONSE'
/** Harness code reported when a quota or rate cap refused the request. */
export const QUOTA_EXCEEDED_CODE = 'QUOTA'

/**
 * Map one provider stop reason onto the harness vocabulary.
 * @param reason - the `stop_reason` Vertex reported, if any.
 * @returns the harness finish reason.
 */
export function mapStopReason(reason: string | undefined): FinishReason {
  switch (reason) {
    case 'end_turn':
    case 'stop_sequence':
    // A server-tool pause cannot recur here: this adapter declares no
    // server-executed tools, so the answer is complete.
    case 'pause_turn':
      return { kind: 'stop' }
    case 'max_tokens':
      return { kind: 'max-tokens' }
    case 'tool_use':
      return { kind: 'tool-calls' }
    case 'refusal':
      return {
        kind: 'error',
        failure: { message: 'google-vertex: the model refused to answer', code: 'REFUSAL' },
      }
    case 'model_context_window_exceeded':
      return {
        kind: 'error',
        failure: {
          message: 'google-vertex: the request exceeded the model context window',
          code: CONTEXT_WINDOW_EXCEEDED_CODE,
        },
      }
    default:
      return { kind: 'stop' }
  }
}

/** Extract the provider's message from a failure body, when it is JSON. */
function failureMessage(body: string): string {
  try {
    const parsed: unknown = JSON.parse(body)
    if (typeof parsed === 'object' && parsed !== null) {
      const error = (parsed as Record<string, unknown>)['error']
      if (typeof error === 'object' && error !== null) {
        const message = (error as Record<string, unknown>)['message']
        if (typeof message === 'string') return message
      }
      const message = (parsed as Record<string, unknown>)['message']
      if (typeof message === 'string') return message
    }
  } catch {
    // Not JSON: the raw body is the most specific thing available.
  }
  return body.slice(0, 300).replace(/\s+/g, ' ').trim()
}

/**
 * Classify a refused HTTP response.
 *
 * Two refusals carry a code of their own because the harness treats them
 * differently: an oversized request must not be retried, and an exhausted quota
 * is a capacity problem rather than an invalid one.
 * @param status - HTTP status.
 * @param body - response body text.
 * @param subject - route description named in the failure.
 * @returns the failure to report.
 */
export function failureForStatus(status: number, body: string, subject: string): LlmFailure {
  const detail = failureMessage(body)
  const message = `google-vertex: ${subject} — HTTP ${status}${detail.length > 0 ? `: ${detail}` : ''}`
  return { message, code: codeForDetail(detail) ?? codeForStatus(status), status }
}

/**
 * Canonical code for one HTTP status, shared by a refused response and by an
 * in-band error envelope naming the same condition numerically.
 * @param status - HTTP status, or the provider's numeric error code.
 * @returns the harness code.
 */
function codeForStatus(status: number): string {
  return status === 400 || status === 413 || status === 422
    ? 'INVALID_REQUEST'
    : status === 401 || status === 403
      ? 'AUTH'
      : status === 404
        ? 'NOT_FOUND'
        : status === 408 || status === 504
          ? 'TIMEOUT'
          : status === 429
            ? 'RATE_LIMIT'
            : status >= 500
              ? 'SERVER'
              : 'TRANSPORT'
}

/**
 * Canonical code for a refusal whose own words name a cause.
 * @param detail - the provider's message or error status text.
 * @returns the code, or undefined when the text names no special cause.
 */
function codeForDetail(detail: string): string | undefined {
  if (/prompt is too long|input (?:is )?too long|maximum context length|context length|too many tokens|exceeds the maximum/i.test(detail)) {
    return CONTEXT_WINDOW_EXCEEDED_CODE
  }
  if (/RESOURCE_EXHAUSTED|quota/i.test(detail)) return QUOTA_EXCEEDED_CODE
  return undefined
}

/** Canonical code for Anthropic's named error type. */
function codeForErrorType(type: string): string {
  return type === 'overloaded_error'
    ? 'SERVER'
    : type === 'rate_limit_error'
      ? 'RATE_LIMIT'
      : type === 'authentication_error' || type === 'permission_error'
        ? 'AUTH'
        : type === 'invalid_request_error'
          ? 'INVALID_REQUEST'
          : 'SERVER'
}

/**
 * Canonical code for Google's `{code, message, status}` error envelope, which
 * classifies exactly as a refused body does: a named cause first, then the
 * numeric code.
 * @param fields - the envelope's own members.
 * @param message - its message, already read.
 * @returns the harness code.
 */
function codeForGoogleError(fields: Record<string, unknown>, message: string): string {
  const status = typeof fields['status'] === 'string' ? fields['status'] : ''
  const detail = codeForDetail(`${status} ${message}`)
  return detail ?? codeForStatus(typeof fields['code'] === 'number' ? fields['code'] : 500)
}

/**
 * Classify a stream-level `error` payload.
 *
 * Both publishers deliver one mid-stream: Anthropic's envelope names the
 * condition in `type`, and Google's in a canonical `status` plus a numeric
 * `code`. A provider error is what turns a mid-stream refusal into a terminal
 * failure instead of a truncated response.
 * @param error - the payload's `error` member.
 * @returns the failure to report.
 */
export function failureForEvent(error: unknown): LlmFailure {
  const fields = (typeof error === 'object' && error !== null ? error : {}) as Record<string, unknown>
  const message = typeof fields['message'] === 'string' ? fields['message'] : 'the provider reported an error'
  const type = fields['type']
  const code = typeof type === 'string'
    ? codeForErrorType(type)
    : codeForGoogleError(fields, message)
  return { message: `google-vertex: ${message}`, code }
}

/**
 * Decode one SSE record (everything up to a blank line) into its JSON payload.
 *
 * `event:` lines are ignored because every Vertex payload carries its own
 * `type`; `data:` lines are concatenated, which is what the SSE specification
 * requires for a multi-line payload.
 * @param record - one raw record, without its trailing blank line.
 * @returns the parsed payload, or undefined for a comment, ping, or malformed record.
 */
export function parseSseRecord(record: string): Record<string, unknown> | undefined {
  const data: string[] = []
  for (const line of record.split('\n')) {
    if (!line.startsWith('data:')) continue
    data.push(line.slice(5).replace(/^ /, ''))
  }
  if (data.length === 0) return undefined
  const payload = data.join('\n')
  if (payload.length === 0) return undefined
  try {
    const parsed: unknown = JSON.parse(payload)
    return typeof parsed === 'object' && parsed !== null ? parsed as Record<string, unknown> : undefined
  } catch {
    return undefined
  }
}

/**
 * Reassembles SSE records from arbitrarily split transport chunks.
 *
 * Line endings are normalized as text arrives, so a `\r\n` split across two
 * chunks still frames one record.
 */
export class SseBuffer {
  #buffer = ''

  /**
   * Absorb one decoded chunk.
   * @param text - newly decoded text.
   * @returns every complete record it completes, in order.
   */
  push(text: string): string[] {
    this.#buffer += text.replace(/\r\n?/g, '\n')
    const records = this.#buffer.split('\n\n')
    this.#buffer = records.pop() ?? ''
    return records
  }

  /**
   * Release whatever a completed body ended with.
   * @returns the trailing record, or undefined when the body ended on a boundary.
   */
  flush(): string | undefined {
    const rest = this.#buffer
    this.#buffer = ''
    return rest.trim().length > 0 ? rest : undefined
  }
}

/** Per-read watchdog hooks the shared pump calls around every outstanding read. */
export interface SsePumpHooks {
  /** Arm the idle bound for the read about to start. */
  armIdle(): void
  /** Clear the armed idle bound; called as soon as a read resolves, and on exit. */
  clearIdle(): void
}

/**
 * Frame one streaming response body into its parsed SSE payloads.
 *
 * Both publisher routes answer with the same framing and differ only in what
 * the payload means, so the read loop, the decoder and record buffers, the
 * end-of-body flush, and the trailing `SseBuffer` record live here once. The
 * adapter supplies the idle watchdog and translates each payload.
 *
 * The idle bound covers one outstanding read: it is armed before every read,
 * cleared as soon as that read resolves — a consumer holding a yielded event is
 * not a stalled provider — and cleared again when the pump exits, whether the
 * body ended, the reader stopped early, or a read threw. A read that throws is
 * left for the adapter to classify.
 * @param body - the response body's byte stream.
 * @param hooks - the caller's idle-watchdog controls.
 * @yields every payload the body framed, in order.
 */
export async function* streamSseRecords(
  body: AsyncIterable<Uint8Array>,
  hooks: SsePumpHooks,
): AsyncGenerator<Record<string, unknown>> {
  const buffer = new SseBuffer()
  const decoder = new TextDecoder()
  /** Parse one record and hand it on when it carried a payload. */
  function* payload(record: string): Generator<Record<string, unknown>> {
    const event = parseSseRecord(record)
    if (event !== undefined) yield event
  }
  try {
    hooks.armIdle()
    for await (const chunk of body) {
      hooks.clearIdle()
      for (const record of buffer.push(decoder.decode(chunk, { stream: true }))) {
        yield * payload(record)
      }
      hooks.armIdle()
    }
    // The decoder holds a partial character and the buffer a partial record;
    // both belong to the body that just ended.
    for (const record of buffer.push(decoder.decode())) yield * payload(record)
    const trailing = buffer.flush()
    if (trailing !== undefined) yield * payload(trailing)
  } finally {
    hooks.clearIdle()
  }
}

/** One content block being assembled from the stream. */
interface PartialBlock {
  blockType: 'text' | 'tool-call'
  text: string
  arguments: string
  id: string
  name: string
  /** True for a provider block this adapter declines to project (thinking, server tools). */
  ignored: boolean
}

/** The index of a stream event, or undefined when absent. */
function eventIndex(event: Record<string, unknown>): number | undefined {
  const index = event['index']
  return typeof index === 'number' && Number.isSafeInteger(index) && index >= 0 ? index : undefined
}

/** A count field, ignoring anything the provider sends that is not a number. */
function count(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

/**
 * Translate Vertex's Anthropic event stream into harness chunks.
 *
 * The translator is stateful because the two protocols disagree about
 * granularity: Vertex announces a content block and then streams deltas, while
 * the harness wants a start, the deltas, and an authoritative close. It is
 * tolerant of the events it does not project — `ping`, thinking, citations,
 * server tool use — because a provider that adds one must not break the turn.
 */
export class StreamTranslator {
  readonly #blocks = new Map<number, PartialBlock>()
  #usage: WireUsage = {}
  #usageReported = false
  #stopReason: string | undefined
  #blocksEmitted = false
  #done = false

  /**
   * Feed one decoded event.
   * @param event - the parsed SSE payload.
   * @returns the chunks this event completes, in order.
   */
  handle(event: Record<string, unknown>): StreamChunk[] {
    if (this.#done) return []
    switch (event['type']) {
      case 'message_start': {
        const message = event['message']
        if (typeof message === 'object' && message !== null) {
          this.#mergeUsage((message as Record<string, unknown>)['usage'])
        }
        return []
      }
      case 'content_block_start':
        return this.#startBlock(event)
      case 'content_block_delta':
        return this.#delta(event)
      case 'content_block_stop': {
        const index = eventIndex(event)
        if (index === undefined) return []
        const partial = this.#blocks.get(index)
        this.#blocks.delete(index)
        if (partial === undefined || partial.ignored) return []
        this.#blocksEmitted = true
        return [{
          type: 'block-end',
          index,
          block: partial.blockType === 'text'
            ? { type: 'text', text: partial.text }
            : {
                type: 'tool-call',
                id: partial.id,
                name: partial.name,
                arguments: partial.arguments.length > 0 ? partial.arguments : '{}',
              },
        }]
      }
      case 'message_delta': {
        const delta = event['delta']
        if (typeof delta === 'object' && delta !== null) {
          const reason = (delta as Record<string, unknown>)['stop_reason']
          if (typeof reason === 'string') this.#stopReason = reason
        }
        this.#mergeUsage(event['usage'])
        return []
      }
      case 'message_stop':
        return this.#finish()
      case 'error':
        this.#done = true
        return [{ type: 'finish', reason: { kind: 'error', failure: failureForEvent(event['error']) } }]
      default:
        return []
    }
  }

  /** Start one content block, or mark it as one this adapter does not project. */
  #startBlock(event: Record<string, unknown>): StreamChunk[] {
    const index = eventIndex(event)
    const raw = event['content_block']
    if (index === undefined || typeof raw !== 'object' || raw === null) return []
    const block = raw as Record<string, unknown>
    switch (block['type']) {
      case 'text':
        this.#blocks.set(index, { blockType: 'text', text: '', arguments: '', id: '', name: '', ignored: false })
        return [{ type: 'block-start', index, blockType: 'text' }]
      case 'tool_use': {
        // Vertex streams a tool call as an empty input object followed by
        // `input_json_delta` fragments; a populated one is a complete call and
        // any later delta would duplicate it.
        const input = block['input']
        const complete = typeof input === 'object' && input !== null && Object.keys(input).length > 0
        const partial: PartialBlock = {
          blockType: 'tool-call',
          text: '',
          arguments: complete ? JSON.stringify(input) : '',
          id: typeof block['id'] === 'string' && block['id'].length > 0 ? block['id'] : `toolu_${index}`,
          name: typeof block['name'] === 'string' ? block['name'] : '',
          ignored: false,
        }
        this.#blocks.set(index, partial)
        return [
          { type: 'block-start', index, blockType: 'tool-call' },
          {
            type: 'tool-call-delta',
            index,
            id: partial.id,
            name: partial.name,
            argumentsDelta: partial.arguments,
          },
        ]
      }
      default:
        this.#blocks.set(index, { blockType: 'text', text: '', arguments: '', id: '', name: '', ignored: true })
        return []
    }
  }

  /** Absorb one content delta. */
  #delta(event: Record<string, unknown>): StreamChunk[] {
    const index = eventIndex(event)
    const raw = event['delta']
    if (index === undefined || typeof raw !== 'object' || raw === null) return []
    const partial = this.#blocks.get(index)
    if (partial === undefined || partial.ignored) return []
    const delta = raw as Record<string, unknown>
    switch (delta['type']) {
      case 'text_delta': {
        const text = typeof delta['text'] === 'string' ? delta['text'] : ''
        if (text.length === 0) return []
        partial.text += text
        return [{ type: 'text-delta', index, text }]
      }
      case 'input_json_delta': {
        const fragment = typeof delta['partial_json'] === 'string' ? delta['partial_json'] : ''
        if (fragment.length === 0) return []
        partial.arguments += fragment
        return [{ type: 'tool-call-delta', index, id: partial.id, argumentsDelta: fragment }]
      }
      default:
        return []
    }
  }

  /** Terminal chunks: usage, then the mapped finish reason. */
  #finish(): StreamChunk[] {
    this.#done = true
    const reason = mapStopReason(this.#stopReason)
    // A completed turn that produced nothing is a degenerate provider response,
    // not a successful empty answer; a provider-reported failure keeps its own
    // reason, which is the more specific account of the same turn.
    if (!this.#blocksEmitted && reason.kind !== 'error') {
      return [{
        type: 'finish',
        reason: {
          kind: 'error',
          failure: {
            message: 'google-vertex: the model completed the response with no content',
            code: EMPTY_RESPONSE_CODE,
          },
        },
      }]
    }
    // Usage accompanies a real provider report; a synthesized zero would claim
    // a measurement that never happened.
    return [
      ...this.#usageReported ? [{ type: 'usage' as const, usage: mapUsage(this.#usage) }] : [],
      { type: 'finish', reason },
    ]
  }

  /** Merge the newest cumulative usage counters. */
  #mergeUsage(raw: unknown): void {
    if (typeof raw !== 'object' || raw === null) return
    const usage = raw as Record<string, unknown>
    const input = count(usage['input_tokens'])
    const output = count(usage['output_tokens'])
    const cacheRead = count(usage['cache_read_input_tokens'])
    const cacheWrite = count(usage['cache_creation_input_tokens'])
    const merged: WireUsage = {
      ...input === undefined ? {} : { input_tokens: input },
      ...output === undefined ? {} : { output_tokens: output },
      ...cacheRead === undefined ? {} : { cache_read_input_tokens: cacheRead },
      ...cacheWrite === undefined ? {} : { cache_creation_input_tokens: cacheWrite },
    }
    // Only a counter the provider actually sent counts as a report.
    if (Object.keys(merged).length === 0) return
    this.#usageReported = true
    this.#usage = { ...this.#usage, ...merged }
  }

  /** True once a terminal event arrived, so the adapter can tell truncation. */
  get done(): boolean {
    return this.#done
  }
}
