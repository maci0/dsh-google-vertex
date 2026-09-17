/**
 * Adapter tests over a stubbed transport: the request that leaves, the chunk
 * contract the harness consumes, and every terminal class — provider refusal,
 * credential failure, truncation, and cancellation.
 *
 * @module dsh-google-vertex/tests/adapter
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { attributionHeaders } from '@deepseek-ai/dsh-llm'

import { GoogleVertexAnthropicAdapter, type VertexAnthropicConfig } from '../src/adapter.ts'
import { VertexAuthError, type FetchLike, type ServiceAccount } from '../src/auth.ts'
import type { GenerateOptions } from '../src/host.ts'
import { DEFAULT_STREAM_IDLE_TIMEOUT_MS } from '../src/wire.ts'
import { collect, stallingResponse, streamResponse, tokens } from './support.ts'

const CONFIG: VertexAnthropicConfig = {
  serviceAccount: { client_email: 'x@y', private_key: 'unused' } as ServiceAccount,
  project: 'p1',
  location: 'global',
  models: [
    { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5 (Vertex)' },
    { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5 (Vertex)' },
  ],
  contextWindow: 200_000,
  maxTokens: 32_000,
  streamIdleTimeoutMs: DEFAULT_STREAM_IDLE_TIMEOUT_MS,
}

const OPTIONS: GenerateOptions = {
  model: 'claude-sonnet-4-5',
  messages: [{ id: '1', role: 'user', content: [{ type: 'text', text: 'hi' }] }],
}

test('streams text through the global publisher endpoint with the bearer token', async () => {
  const requests: { url: string; init: RequestInit }[] = []
  const fetch: FetchLike = (url, init) => {
    requests.push({ url, init })
    return Promise.resolve(streamResponse([
      'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":11,"cache_read_input_tokens":2}}}\n\n',
      'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
      'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n',
      'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]))
  }
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hi' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hi' } },
    { type: 'usage', usage: { inputTokens: 11, outputTokens: 2, totalTokens: 15, cacheReadTokens: 2 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])

  const request = requests[0]
  assert.equal(
    request?.url,
    'https://aiplatform.googleapis.com/v1/projects/p1/locations/global/publishers/anthropic/models/claude-sonnet-4-5:streamRawPredict',
  )
  const headers = request?.init.headers as Record<string, string>
  assert.equal(headers['authorization'], 'Bearer access-token')
  // Assert against the live helper, never a pinned copy that could drift.
  const userAgent = String(attributionHeaders()['user-agent'])
  assert.equal(headers['user-agent'], userAgent)
  // The idle watchdog's combined signal is what the request carries, so a
  // stalled read can be torn down.
  assert.equal(request?.init.signal instanceof AbortSignal, true)
  assert.match(userAgent, /^deepseek-harness\/\S+ \(\+https:\/\/github\.com\/deepseek-ai\/deepseek-harness\)$/)

  const body = JSON.parse(String(request?.init.body)) as Record<string, unknown>
  assert.equal(body['stream'], true)
  assert.equal(body['anthropic_version'], 'vertex-2023-10-16')
  // Vertex names the model in the path, not the body.
  assert.equal('model' in body, false)
  assert.equal(body['max_tokens'], 32_000)
})

test('a tool turn reaches the harness as raw JSON argument deltas', async () => {
  const fetch: FetchLike = () => Promise.resolve(streamResponse([
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_9","name":"read","input":{}}}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"a"}}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"}"}}\n\n',
    'data: {"type":"content_block_stop","index":0}\n\n',
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}\n\n',
    'data: {"type":"message_stop"}\n\n',
  ]))
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.deepEqual(chunks.at(-2), { type: 'usage', usage: { inputTokens: 0, outputTokens: 9, totalTokens: 9 } })
  assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'tool-calls' } })
  assert.deepEqual(chunks[0], { type: 'block-start', index: 0, blockType: 'tool-call' })
  const closed = chunks.find(chunk => chunk.type === 'block-end')
  assert.deepEqual(closed, {
    type: 'block-end',
    index: 0,
    block: { type: 'tool-call', id: 'toolu_9', name: 'read', arguments: '{"path":"a"}' },
  })
})

test('a refused request becomes one classified terminal finish', async () => {
  const fetch: FetchLike = () => Promise.resolve(new Response(
    JSON.stringify({ error: { code: 404, message: 'Publisher Model `claude-nope` is not servable in region global.', status: 'NOT_FOUND' } }),
    { status: 404, headers: { 'content-type': 'application/json' } },
  ))
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.equal(chunks.length, 1)
  assert.deepEqual(chunks[0], {
    type: 'finish',
    reason: {
      kind: 'error',
      failure: {
        message: 'google-vertex: model "claude-sonnet-4-5" in global — HTTP 404: Publisher Model `claude-nope` is not servable in region global.',
        code: 'NOT_FOUND',
        status: 404,
      },
    },
  })
})

test('a credential failure never reaches the wire', async () => {
  let called = 0
  const fetch: FetchLike = () => {
    called += 1
    return Promise.resolve(streamResponse([]))
  }
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, {
    fetch,
    tokens: { get: () => Promise.reject(new VertexAuthError('AUTH', 'google-vertex: token endpoint refused the service account (HTTP 400)')) },
  })
  const chunks = await collect(adapter, OPTIONS)

  assert.equal(called, 0)
  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: {
      kind: 'error',
      failure: { message: 'google-vertex: token endpoint refused the service account (HTTP 400)', code: 'AUTH' },
    },
  }])
})

test('a body that ends before message_stop is a transport truncation', async () => {
  const fetch: FetchLike = () => Promise.resolve(streamResponse([
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}\n\n',
  ]))
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.deepEqual(chunks.at(-1), {
    type: 'finish',
    reason: {
      kind: 'error',
      failure: {
        message: 'google-vertex: model "claude-sonnet-4-5" stream ended before message_stop',
        code: 'TRANSPORT',
      },
    },
  })
})

test('an aborted request reports cancellation, not a transport fault', async () => {
  const controller = new AbortController()
  const fetch: FetchLike = () => {
    controller.abort()
    return Promise.reject(new Error('The operation was aborted'))
  }
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, { ...OPTIONS, signal: controller.signal })

  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: { kind: 'aborted', failure: { message: 'google-vertex: request aborted', code: 'ABORTED' } },
  }])
})

test('one token serves many streams', async () => {
  const source = tokens()
  const fetch: FetchLike = () => Promise.resolve(streamResponse(['data: {"type":"message_stop"}\n\n']))
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, { fetch, tokens: source })
  await collect(adapter, OPTIONS)
  await collect(adapter, OPTIONS)
  assert.equal(source.calls(), 2)
})

test('a stalled stream is one TIMEOUT finish, not a hang', async () => {
  const fetch: FetchLike = (_url, init) => Promise.resolve(stallingResponse(init?.signal, [
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}\n\n',
  ]))
  const adapter = new GoogleVertexAnthropicAdapter({ ...CONFIG, streamIdleTimeoutMs: 20 }, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  // The bytes that did arrive are kept, and exactly one terminal chunk reports
  // why the rest never did.
  assert.deepEqual(chunks.slice(0, 2), [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'partial' },
  ])
  const finishes = chunks.filter(chunk => chunk.type === 'finish')
  assert.equal(finishes.length, 1)
  assert.deepEqual(finishes[0], {
    type: 'finish',
    reason: {
      kind: 'error',
      failure: { message: 'google-vertex: no stream data for 20ms (streamIdleTimeoutMs)', code: 'TIMEOUT' },
    },
  })
})

test('a hung token mint is bounded by the same watchdog signal', async () => {
  const signals: (AbortSignal | undefined)[] = []
  const adapter = new GoogleVertexAnthropicAdapter({ ...CONFIG, streamIdleTimeoutMs: 20 }, {
    fetch: () => Promise.reject(new Error('the request must never leave')),
    tokens: {
      get: (signal?: AbortSignal) => {
        signals.push(signal)
        return new Promise<string>((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new Error('The operation was aborted')), { once: true })
        })
      },
    },
  })
  const chunks = await collect(adapter, OPTIONS)

  assert.equal(signals.length, 1)
  assert.equal(signals[0]?.aborted, true)
  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: {
      kind: 'error',
      failure: { message: 'google-vertex: no stream data for 20ms (streamIdleTimeoutMs)', code: 'TIMEOUT' },
    },
  }])
})

test('an in-band error followed by a stalled body still yields one terminal chunk', async () => {
  const fetch: FetchLike = (_url, init) => Promise.resolve(stallingResponse(init?.signal, [
    'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n',
  ]))
  const adapter = new GoogleVertexAnthropicAdapter({ ...CONFIG, streamIdleTimeoutMs: 20 }, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: { kind: 'error', failure: { message: 'google-vertex: Overloaded', code: 'SERVER' } },
  }])
})

test('a stream that reported no counters emits no usage chunk', async () => {
  const fetch: FetchLike = () => Promise.resolve(streamResponse([
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n',
    'data: {"type":"content_block_stop","index":0}\n\n',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n',
    'data: {"type":"message_stop"}\n\n',
  ]))
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.equal(chunks.some(chunk => chunk.type === 'usage'), false)
  assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'stop' } })
})

test('model metadata is the configured catalog plus the resolved capacities', async () => {
  const adapter = new GoogleVertexAnthropicAdapter(CONFIG, { fetch: globalThis.fetch as FetchLike, tokens: tokens() })

  const listed = await adapter.listModels('google-vertex-anthropic')
  assert.deepEqual(listed.map(model => model.id), ['claude-sonnet-4-5', 'claude-haiku-4-5'])
  assert.deepEqual(listed[0]?.inputModalities, ['text'])

  const resolved = await adapter.resolveModel('google-vertex-anthropic', 'claude-sonnet-4-5')
  assert.equal(resolved.name, 'Claude Sonnet 4.5 (Vertex)')
  assert.deepEqual(resolved.context, { contextWindow: 200_000 })
  assert.equal(resolved.defaultMaxTokens, 32_000)

  // An unlisted id is accepted, named by itself: catalog membership is advisory.
  const unknown = await adapter.resolveModel('google-vertex-anthropic', 'claude-future')
  assert.equal(unknown.name, 'claude-future')

  assert.deepEqual(adapter.providerInfo('google-vertex-anthropic'), {
    id: 'google-vertex-anthropic',
    name: 'Google Vertex AI (Anthropic)',
  })
  assert.equal(adapter.providerRetryPolicy('google-vertex-anthropic'), undefined)
  assert.equal(adapter.imageRequestPricing('google-vertex-anthropic', 'claude-future'), undefined)
})
