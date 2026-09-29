/**
 * Regression: the harness hands a tool result to an adapter as a role-`tool`
 * message built by `createToolResultMessage` (call id and error flag on the
 * message), not as a `tool-result` content block inside a user turn. Both
 * publisher routes must project that message into the provider's own result
 * envelope, or the provider refuses the next turn: Anthropic demands a
 * `tool_result` for every `tool_use`, and Gemini a `functionResponse` for every
 * `functionCall`.
 *
 * @module dsh-google-vertex/tests/tool-round-trip
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { createToolResultMessage, ToolCallId } from '@deepseek-ai/dsh-llm'

import { GoogleVertexAnthropicAdapter, type VertexAnthropicConfig } from '../src/adapter.ts'
import type { FetchLike, ServiceAccount } from '../src/auth.ts'
import { buildGeminiRequest, DEFAULT_GEMINI_MODELS } from '../src/gemini.ts'
import { GoogleVertexGeminiAdapter, type GeminiAdapterConfig } from '../src/gemini_adapter.ts'
import type { GenerateOptions } from '../src/host.ts'
import { buildRequestBody, DEFAULT_STREAM_IDLE_TIMEOUT_MS, type VertexWireConfig } from '../src/wire.ts'
import { collect, streamResponse, tokens } from './support.ts'

const WIRE: VertexWireConfig = { project: 'p1', location: 'global', maxTokens: 32_000 }

const ANTHROPIC_CONFIG: VertexAnthropicConfig = {
  serviceAccount: { client_email: 'x@y', private_key: 'unused' } as ServiceAccount,
  project: 'p1',
  location: 'global',
  models: [{ id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5 (Vertex)' }],
  contextWindow: 200_000,
  maxTokens: 32_000,
  streamIdleTimeoutMs: DEFAULT_STREAM_IDLE_TIMEOUT_MS,
}

const GEMINI_CONFIG: GeminiAdapterConfig = {
  serviceAccount: { client_email: 'x@y', private_key: 'unused' } as ServiceAccount,
  project: 'p1',
  location: 'global',
  models: DEFAULT_GEMINI_MODELS,
  maxTokens: 65_535,
  streamIdleTimeoutMs: DEFAULT_STREAM_IDLE_TIMEOUT_MS,
}

/**
 * The exact message the harness itself produces for a tool result. The cast is
 * only for the mirror's narrower structural content type; the runtime shape is
 * the harness's.
 */
function harnessToolResult(callId: string, text: string, isError = false): GenerateOptions['messages'][number] {
  return createToolResultMessage({
    callId: ToolCallId(callId),
    content: [{ type: 'text', text }],
    isError,
  }) as unknown as GenerateOptions['messages'][number]
}

/** One assistant tool call followed by its harness tool result. */
function toolRoundTrip(model: string, text = '18C and clear', isError = false): GenerateOptions {
  return {
    model,
    messages: [
      { id: '1', role: 'user', content: [{ type: 'text', text: 'weather in Paris?' }] },
      { id: '2', role: 'assistant', content: [{ type: 'tool-call', id: 'call_1', name: 'get_weather', arguments: '{"city":"Paris"}' }] },
      harnessToolResult('call_1', text, isError),
    ],
  }
}

test('a harness tool message becomes the tool_result block Anthropic requires', () => {
  const body = buildRequestBody(toolRoundTrip('claude-sonnet-4-5'), WIRE)

  // The last block of the conversation carries the one message-level cache
  // breakpoint; the result itself must be a tool_result naming the call.
  assert.deepEqual(body.messages[2], {
    role: 'user',
    content: [{
      type: 'tool_result',
      tool_use_id: 'call_1',
      content: '18C and clear',
      cache_control: { type: 'ephemeral' },
    }],
  })

  const failed = buildRequestBody(toolRoundTrip('claude-sonnet-4-5', 'boom', true), WIRE)
  assert.deepEqual(failed.messages.at(-1)?.content[0], {
    type: 'tool_result',
    tool_use_id: 'call_1',
    content: 'boom',
    is_error: true,
    cache_control: { type: 'ephemeral' },
  })
})

test('a harness tool message becomes the functionResponse part Gemini requires', () => {
  const body = buildGeminiRequest(toolRoundTrip('gemini-3.5-flash'), { ...WIRE, maxTokens: 65_535 })

  assert.deepEqual(body.contents[2], {
    role: 'user',
    parts: [{
      functionResponse: {
        name: 'get_weather',
        id: 'call_1',
        response: { result: '18C and clear' },
      },
    }],
  })
})

test('the Anthropic route sends the tool_result on the wire', async () => {
  const bodies: Record<string, unknown>[] = []
  const fetch: FetchLike = (_url, init) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
    return Promise.resolve(streamResponse(['data: {"type":"message_stop"}\n\n']))
  }
  const adapter = new GoogleVertexAnthropicAdapter(ANTHROPIC_CONFIG, { fetch, tokens: tokens() })
  await collect(adapter, toolRoundTrip('claude-sonnet-4-5'))

  assert.deepEqual(bodies[0]?.['messages'], [
    { role: 'user', content: [{ type: 'text', text: 'weather in Paris?' }] },
    { role: 'assistant', content: [{ type: 'tool_use', id: 'call_1', name: 'get_weather', input: { city: 'Paris' } }] },
    {
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: 'call_1',
        content: '18C and clear',
        cache_control: { type: 'ephemeral' },
      }],
    },
  ])
})

test('the Gemini route sends the functionResponse on the wire', async () => {
  const bodies: Record<string, unknown>[] = []
  const fetch: FetchLike = (_url, init) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
    return Promise.resolve(streamResponse(['data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}\n\n']))
  }
  const adapter = new GoogleVertexGeminiAdapter(GEMINI_CONFIG, { fetch, tokens: tokens() })
  await collect(adapter, toolRoundTrip('gemini-3.5-flash'))

  assert.deepEqual(bodies[0]?.['contents'], [
    { role: 'user', parts: [{ text: 'weather in Paris?' }] },
    { role: 'model', parts: [{ functionCall: { name: 'get_weather', args: { city: 'Paris' }, id: 'call_1' } }] },
    {
      role: 'user',
      parts: [{
        functionResponse: {
          name: 'get_weather',
          id: 'call_1',
          response: { result: '18C and clear' },
        },
      }],
    },
  ])
})
