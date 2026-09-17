/**
 * Gemini adapter tests over a stubbed transport: the request that leaves (with
 * the replayed thought signature), the chunk contract, and the terminal classes.
 *
 * @module dsh-google-vertex/tests/gemini-adapter
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { attributionHeaders } from '@deepseek-ai/dsh-llm'

import { VertexAuthError, type FetchLike, type ServiceAccount } from '../src/auth.ts'
import { DEFAULT_GEMINI_MODELS } from '../src/gemini.ts'
import { GoogleVertexGeminiAdapter, type GeminiAdapterConfig } from '../src/gemini_adapter.ts'
import type { GenerateOptions, Message } from '../src/host.ts'
import { DEFAULT_STREAM_IDLE_TIMEOUT_MS } from '../src/wire.ts'
import { collect, stallingResponse, streamResponse, tokens } from './support.ts'

const CONFIG: GeminiAdapterConfig = {
  serviceAccount: { client_email: 'x@y', private_key: 'unused' } as ServiceAccount,
  project: 'p1',
  location: 'global',
  models: DEFAULT_GEMINI_MODELS,
  maxTokens: 65_535,
  streamIdleTimeoutMs: DEFAULT_STREAM_IDLE_TIMEOUT_MS,
}

const MODEL = 'gemini-3.5-flash'
const OPTIONS: GenerateOptions = {
  model: MODEL,
  messages: [{ id: '1', role: 'user', content: [{ type: 'text', text: 'hi' }] }],
}

test('streams Gemini text through the publisher endpoint with the bearer token', async () => {
  const requests: { url: string; init: RequestInit }[] = []
  const fetch: FetchLike = (url, init) => {
    requests.push({ url, init })
    return Promise.resolve(streamResponse([
      'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}]}\n\n',
      'data: {"candidates":[{"content":{"role":"model","parts":[{"text":" there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":2,"totalTokenCount":13}}\n\n',
    ]))
  }
  const adapter = new GoogleVertexGeminiAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hi' },
    { type: 'text-delta', index: 0, text: ' there' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hi there' } },
    { type: 'usage', usage: { inputTokens: 11, outputTokens: 2, totalTokens: 13 } },
    {
      type: 'finish',
      reason: { kind: 'stop' },
      replayState: { response: { kind: 'google-vertex-gemini', version: 1, model: MODEL }, blocks: [{ type: 'text' }] },
    },
  ])

  const request = requests[0]
  assert.equal(
    request?.url,
    'https://aiplatform.googleapis.com/v1/projects/p1/locations/global/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse',
  )
  const headers = request?.init.headers as Record<string, string>
  assert.equal(headers['authorization'], 'Bearer access-token')
  // Assert against the live helper, never a pinned copy that could drift.
  const userAgent = String(attributionHeaders()['user-agent'])
  assert.equal(headers['user-agent'], userAgent)
  assert.match(userAgent, /^deepseek-harness\/\S+ \(\+https:\/\/github\.com\/deepseek-ai\/deepseek-harness\)$/)
  const body = JSON.parse(String(request?.init.body)) as Record<string, unknown>
  assert.equal('model' in body, false)
})

test('a replayed tool call sends the thought signature Vertex demands', async () => {
  const bodies: Record<string, unknown>[] = []
  const fetch: FetchLike = (_url, init) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
    return Promise.resolve(streamResponse([
      'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"18C and clear"}]},"finishReason":"STOP"}]}\n\n',
    ]))
  }
  const assistant: Message = {
    id: '2',
    role: 'assistant',
    content: [{ type: 'tool-call', id: 'call_1', name: 'get_weather', arguments: '{"city":"Paris"}' }],
    source: {
      kind: 'model',
      replayState: {
        response: { kind: 'google-vertex-gemini', version: 1, model: MODEL },
        blocks: [{ type: 'tool-call', thoughtSignature: 'sig-abc' }],
      },
    },
  }
  const adapter = new GoogleVertexGeminiAdapter(CONFIG, { fetch, tokens: tokens() })
  await collect(adapter, {
    model: MODEL,
    messages: [
      { id: '1', role: 'user', content: [{ type: 'text', text: 'weather?' }] },
      assistant,
      { id: '3', role: 'user', content: [{ type: 'tool-result', toolCallId: 'call_1', content: [{ type: 'text', text: '18C and clear' }] }] },
    ],
  })

  const contents = bodies[0]?.['contents'] as { role: string; parts: Record<string, unknown>[] }[]
  assert.deepEqual(contents[1]?.parts[0], {
    functionCall: { name: 'get_weather', args: { city: 'Paris' }, id: 'call_1' },
    thoughtSignature: 'sig-abc',
  })
  assert.deepEqual(contents[2]?.parts[0], {
    functionResponse: { name: 'get_weather', id: 'call_1', response: { result: '18C and clear' } },
  })
})

test('a refused request becomes one classified terminal finish', async () => {
  const fetch: FetchLike = () => Promise.resolve(new Response(JSON.stringify({
    error: { code: 404, message: 'Publisher Model `gemini-nope` was not found or your project does not have access to it.', status: 'NOT_FOUND' },
  }), { status: 404, headers: { 'content-type': 'application/json' } }))
  const adapter = new GoogleVertexGeminiAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.equal(chunks.length, 1)
  const finish = chunks[0]
  assert.equal(finish?.type, 'finish')
  assert.equal(finish?.type === 'finish' && finish.reason.kind === 'error' ? finish.reason.failure.code : '', 'NOT_FOUND')
  assert.equal(finish?.type === 'finish' && finish.reason.kind === 'error' ? finish.reason.failure.status : undefined, 404)
})

test('a body that ends without a finish reason is a transport truncation', async () => {
  const fetch: FetchLike = () => Promise.resolve(streamResponse([
    'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}\n\n',
  ]))
  const adapter = new GoogleVertexGeminiAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.deepEqual(chunks.at(-1), {
    type: 'finish',
    reason: {
      kind: 'error',
      failure: { message: `google-vertex: model "${MODEL}" stream ended before a finish reason`, code: 'TRANSPORT' },
    },
  })
})

test('a credential failure never reaches the wire, and abort reports cancellation', async () => {
  let called = 0
  const fetch: FetchLike = () => {
    called += 1
    return Promise.resolve(streamResponse([]))
  }
  const failing = new GoogleVertexGeminiAdapter(CONFIG, {
    fetch,
    tokens: { get: () => Promise.reject(new VertexAuthError('AUTH', 'google-vertex: token endpoint refused the service account (HTTP 400)')) },
  })
  assert.deepEqual(await collect(failing, OPTIONS), [{
    type: 'finish',
    reason: {
      kind: 'error',
      failure: { message: 'google-vertex: token endpoint refused the service account (HTTP 400)', code: 'AUTH' },
    },
  }])
  assert.equal(called, 0)

  const controller = new AbortController()
  const aborting = new GoogleVertexGeminiAdapter(CONFIG, {
    fetch: () => {
      controller.abort()
      return Promise.reject(new Error('The operation was aborted'))
    },
    tokens: tokens(),
  })
  assert.deepEqual(await collect(aborting, { ...OPTIONS, signal: controller.signal }), [{
    type: 'finish',
    reason: { kind: 'aborted', failure: { message: 'google-vertex: request aborted', code: 'ABORTED' } },
  }])
})

test('an in-band provider error is the one terminal chunk, with the provider code', async () => {
  const cases: { payload: string; message: string; code: string }[] = [
    {
      payload: '{"error":{"code":429,"message":"Rate limit exceeded for model gemini-3.5-flash."}}',
      message: 'Rate limit exceeded for model gemini-3.5-flash.',
      code: 'RATE_LIMIT',
    },
    {
      payload: '{"error":{"code":503,"message":"The service is currently unavailable."}}',
      message: 'The service is currently unavailable.',
      code: 'SERVER',
    },
    {
      payload: '{"error":{"code":429,"message":"Resource exhausted.","status":"RESOURCE_EXHAUSTED"}}',
      message: 'Resource exhausted.',
      code: 'QUOTA',
    },
  ]
  for (const { payload, message, code } of cases) {
    const fetch: FetchLike = () => Promise.resolve(streamResponse([`data: ${payload}\n\n`]))
    const adapter = new GoogleVertexGeminiAdapter(CONFIG, { fetch, tokens: tokens() })
    const chunks = await collect(adapter, OPTIONS)

    // The error is what ends the stream: no truncation report follows it, and
    // the provider's own message survives.
    assert.equal(chunks.length, 1)
    assert.deepEqual(chunks[0], {
      type: 'finish',
      reason: { kind: 'error', failure: { message: `google-vertex: ${message}`, code } },
    })
  }
})

test('a stalled stream is one TIMEOUT finish, not a hang', async () => {
  const fetch: FetchLike = (_url, init) => Promise.resolve(stallingResponse(init?.signal, [
    'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}\n\n',
  ]))
  const adapter = new GoogleVertexGeminiAdapter({ ...CONFIG, streamIdleTimeoutMs: 20 }, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

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

test('an in-band error followed by a stalled body still yields one terminal chunk', async () => {
  const fetch: FetchLike = (_url, init) => Promise.resolve(stallingResponse(init?.signal, [
    'data: {"error":{"code":503,"message":"The service is currently unavailable."}}\n\n',
  ]))
  const adapter = new GoogleVertexGeminiAdapter({ ...CONFIG, streamIdleTimeoutMs: 20 }, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: {
      kind: 'error',
      failure: { message: 'google-vertex: The service is currently unavailable.', code: 'SERVER' },
    },
  }])
})

test('a stream that reported no counters emits no usage chunk', async () => {
  const fetch: FetchLike = () => Promise.resolve(streamResponse([
    'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]},"finishReason":"STOP"}]}\n\n',
  ]))
  const adapter = new GoogleVertexGeminiAdapter(CONFIG, { fetch, tokens: tokens() })
  const chunks = await collect(adapter, OPTIONS)

  assert.equal(chunks.some(chunk => chunk.type === 'usage'), false)
  assert.equal(chunks.at(-1)?.type, 'finish')
})

test('model metadata comes from the catalog and reports the family capacities', async () => {
  const adapter = new GoogleVertexGeminiAdapter(CONFIG, { fetch: globalThis.fetch as FetchLike, tokens: tokens() })

  const listed = await adapter.listModels('google-vertex-gemini')
  assert.equal(listed.length, DEFAULT_GEMINI_MODELS.length)
  assert.deepEqual(listed[0]?.inputModalities, ['text'])

  const resolved = await adapter.resolveModel('google-vertex-gemini', 'gemini-2.5-flash-lite')
  assert.equal(resolved.name, 'Gemini 2.5 Flash-Lite (Vertex)')
  // One below Vertex's exclusive ceiling: 65536 is refused outright.
  assert.equal(resolved.defaultMaxTokens, 65_535)
  assert.deepEqual(resolved.context, { contextWindow: 1_048_576 })

  const unknown = await adapter.resolveModel('google-vertex-gemini', 'gemini-9')
  assert.equal(unknown.name, 'gemini-9')
  assert.equal(unknown.defaultMaxTokens, 65_535)

  assert.deepEqual(adapter.providerInfo('google-vertex-gemini'), {
    id: 'google-vertex-gemini',
    name: 'Google Vertex AI (Gemini)',
  })
  assert.equal(adapter.providerRetryPolicy('google-vertex-gemini'), undefined)
  assert.equal(adapter.imageRequestPricing('google-vertex-gemini', 'gemini-9'), undefined)
})
