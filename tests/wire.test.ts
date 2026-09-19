/**
 * Pure wire tests: request projection, cache breakpoints, SSE framing, usage
 * accounting, stop-reason mapping, and failure classification.
 *
 * @module dsh-google-vertex/tests/wire
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import type { GenerateOptions } from '../src/host.ts'
import {
  buildRequestBody,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  endpointFor,
  failureForEvent,
  failureForStatus,
  mapStopReason,
  mapUsage,
  parseSseRecord,
  QUOTA_EXCEEDED_CODE,
  SseBuffer,
  StreamTranslator,
  type VertexWireConfig,
  type WireMessage,
} from '../src/wire.ts'

const CONFIG: VertexWireConfig = { project: 'p1', location: 'global', maxTokens: 1234 }

/** The last block of one projected wire message. */
function lastBlock(message: WireMessage | undefined): Record<string, unknown> | undefined {
  return message?.content.at(-1) as Record<string, unknown> | undefined
}

test('endpointFor keeps the global host unprefixed and regions prefixed', () => {
  assert.equal(
    endpointFor('p1', 'global', 'claude-sonnet-4-5'),
    'https://aiplatform.googleapis.com/v1/projects/p1/locations/global/publishers/anthropic/models/claude-sonnet-4-5:streamRawPredict',
  )
  assert.equal(
    endpointFor('p1', 'us-east5', 'claude-sonnet-4-5'),
    'https://us-east5-aiplatform.googleapis.com/v1/projects/p1/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5:streamRawPredict',
  )
})

test('buildRequestBody hoists system text, merges roles, and maps tool blocks', () => {
  const options: GenerateOptions = {
    model: 'claude-sonnet-4-5',
    system: 'one-shot system',
    maxTokens: undefined,
    temperature: 0.5,
    stop: ['STOP'],
    tools: [{ name: 'read', description: 'read a file', parameters: { properties: { path: { type: 'string' } } } }],
    messages: [
      { id: '0', role: 'system', content: [{ type: 'text', text: 'loop system' }] },
      { id: '1', role: 'user', content: [{ type: 'text', text: 'hi' }] },
      { id: '2', role: 'assistant', content: [{ type: 'reasoning', text: 'thinking' }, { type: 'text', text: 'sure' }] },
      { id: '3', role: 'assistant', content: [{ type: 'tool-call', id: 'tu_1', name: 'read', arguments: '{"path":"a.txt"}' }] },
      { id: '4', role: 'user', content: [{ type: 'tool-result', toolCallId: 'tu_1', content: [{ type: 'text', text: 'contents' }] }] },
      { id: '5', role: 'user', content: [{ type: 'text', text: 'and now?' }] },
    ],
  }
  const body = buildRequestBody(options, CONFIG)

  assert.equal(body.anthropic_version, 'vertex-2023-10-16')
  assert.equal(body.stream, true)
  assert.equal(body.max_tokens, 1234)
  assert.equal(body.temperature, 0.5)
  assert.deepEqual(body.stop_sequences, ['STOP'])
  assert.deepEqual(body.system?.map(block => block.text), ['one-shot system', 'loop system'])

  // Reasoning dropped, the two assistant turns merged into one.
  assert.deepEqual(body.messages.map(message => message.role), ['user', 'assistant', 'user'])
  assert.deepEqual(body.messages[1]?.content.map(block => block.type), ['text', 'tool_use'])
  assert.deepEqual(body.messages[2]?.content.map(block => block.type), ['tool_result', 'text'])
  assert.deepEqual(body.messages[2]?.content[0], {
    type: 'tool_result',
    tool_use_id: 'tu_1',
    content: 'contents',
  })
  // The one message-level breakpoint is the conversation's final block.
  assert.deepEqual(lastBlock(body.messages[2])?.['cache_control'], { type: 'ephemeral' })

  // A tool schema without an explicit type is completed for the provider.
  assert.deepEqual(body.tools?.[0]?.input_schema['type'], 'object')
  assert.equal(body.tools?.[0]?.input_schema['properties'] !== undefined, true)
})

test('cache breakpoints land on tools, system, and the final block', () => {
  const body = buildRequestBody({
    model: 'm',
    system: 'sys',
    tools: [
      { name: 'a', description: 'a', parameters: {} },
      { name: 'b', description: 'b', parameters: {} },
    ],
    messages: [{ id: '1', role: 'user', content: [{ type: 'text', text: 'hi' }] }],
  }, CONFIG)

  assert.equal(body.tools?.[0]?.cache_control, undefined)
  assert.deepEqual(body.tools?.[1]?.cache_control, { type: 'ephemeral' })
  assert.deepEqual(body.system?.[0]?.cache_control, { type: 'ephemeral' })
  assert.deepEqual(lastBlock(body.messages[0])?.['cache_control'], { type: 'ephemeral' })
})

test('an empty tool result is still a result the provider accepts', () => {
  const body = buildRequestBody({
    model: 'm',
    messages: [{ id: '1', role: 'user', content: [{ type: 'tool-result', toolCallId: 'tu', content: [] }] }],
  }, CONFIG)
  assert.equal((body.messages[0]?.content[0] as Record<string, unknown> | undefined)?.['content'], '(no output)')
})

test('mapUsage keeps harness counts disjoint and totals them', () => {
  assert.deepEqual(mapUsage({ input_tokens: 10, output_tokens: 4, cache_read_input_tokens: 100, cache_creation_input_tokens: 5 }), {
    inputTokens: 10,
    outputTokens: 4,
    totalTokens: 119,
    cacheReadTokens: 100,
    cacheWriteTokens: 5,
  })
  assert.deepEqual(mapUsage({ input_tokens: 3, output_tokens: 1 }), {
    inputTokens: 3,
    outputTokens: 1,
    totalTokens: 4,
  })
})

test('mapStopReason covers tool use, truncation, refusal, and overflow', () => {
  assert.deepEqual(mapStopReason('end_turn'), { kind: 'stop' })
  assert.deepEqual(mapStopReason('stop_sequence'), { kind: 'stop' })
  assert.deepEqual(mapStopReason(undefined), { kind: 'stop' })
  assert.deepEqual(mapStopReason('max_tokens'), { kind: 'max-tokens' })
  assert.deepEqual(mapStopReason('tool_use'), { kind: 'tool-calls' })
  assert.deepEqual(mapStopReason('model_context_window_exceeded'), {
    kind: 'error',
    failure: {
      message: 'google-vertex: the request exceeded the model context window',
      code: CONTEXT_WINDOW_EXCEEDED_CODE,
    },
  })
  assert.equal(mapStopReason('refusal').kind, 'error')
})

test('failureForStatus classifies the refusals the harness treats differently', () => {
  const notFound = failureForStatus(404, JSON.stringify({
    error: { code: 404, message: 'Publisher Model `publishers/anthropic/models/x` is not servable in region us-central1.', status: 'NOT_FOUND' },
  }), 'model "x" in us-central1')
  assert.equal(notFound.code, 'NOT_FOUND')
  assert.match(notFound.message, /not servable in region us-central1/)

  assert.equal(failureForStatus(401, '', 'x').code, 'AUTH')
  assert.equal(failureForStatus(429, '', 'x').code, 'RATE_LIMIT')
  assert.equal(failureForStatus(503, '', 'x').code, 'SERVER')
  assert.equal(failureForStatus(400, '{"error":{"message":"thinking.enabled.budget_tokens: Input should be greater than or equal to 1024"}}', 'x').code, 'INVALID_REQUEST')
  assert.equal(failureForStatus(429, '{"error":{"message":"Quota exceeded for aiplatform.googleapis.com/global_online_prediction_tokens_per_minute_per_project"}}', 'x').code, QUOTA_EXCEEDED_CODE)
  assert.equal(failureForStatus(400, '{"error":{"message":"prompt is too long: 210000 tokens > 200000 maximum"}}', 'x').code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('failureForEvent classifies both publishers’ error envelopes', () => {
  // Anthropic's named types, unchanged.
  assert.equal(failureForEvent({ type: 'overloaded_error', message: 'Overloaded' }).code, 'SERVER')
  assert.equal(failureForEvent({ type: 'rate_limit_error', message: 'slow down' }).code, 'RATE_LIMIT')
  assert.equal(failureForEvent({ type: 'authentication_error', message: 'bad key' }).code, 'AUTH')
  assert.equal(failureForEvent({ type: 'invalid_request_error', message: 'nope' }).code, 'INVALID_REQUEST')

  // Google's numeric code and canonical status.
  assert.equal(failureForEvent({ code: 429, message: 'Rate limit exceeded' }).code, 'RATE_LIMIT')
  assert.equal(failureForEvent({ code: 503, message: 'unavailable' }).code, 'SERVER')
  assert.equal(failureForEvent({ code: 401, message: 'unauthenticated' }).code, 'AUTH')
  assert.equal(failureForEvent({ code: 404, message: 'not found' }).code, 'NOT_FOUND')
  assert.equal(failureForEvent({ code: 408, message: 'deadline' }).code, 'TIMEOUT')
  assert.equal(failureForEvent({ code: 400, message: 'bad argument' }).code, 'INVALID_REQUEST')
  assert.equal(failureForEvent({ code: 429, message: 'Resource exhausted.', status: 'RESOURCE_EXHAUSTED' }).code, QUOTA_EXCEEDED_CODE)
  assert.equal(failureForEvent({}).code, 'SERVER')
  assert.match(failureForEvent({ code: 429, message: 'slow down' }).message, /^google-vertex: slow down$/)
})

test('SseBuffer frames records split across transport chunks', () => {
  const buffer = new SseBuffer()
  assert.deepEqual(buffer.push('event: ping\ndata: {"ty'), [])
  assert.deepEqual(buffer.push('pe":"ping"}\n\n'), ['event: ping\ndata: {"type":"ping"}'])
  // CRLF split across two chunks still frames one record.
  assert.deepEqual(buffer.push('data: {"type":"message_start","message":{"usage":{"input_tokens":7}}}\r'), [])
  assert.deepEqual(buffer.push('\n\r\n'), ['data: {"type":"message_start","message":{"usage":{"input_tokens":7}}}'])
  assert.equal(buffer.flush(), undefined)
})

test('parseSseRecord joins multi-line data and ignores non-data lines', () => {
  assert.deepEqual(parseSseRecord('event: content_block_delta\ndata: {"type":"content_block_delta"}'), {
    type: 'content_block_delta',
  })
  assert.deepEqual(parseSseRecord('data: {"a":1,\ndata: "b":2}'), { a: 1, b: 2 })
  assert.equal(parseSseRecord(': keepalive'), undefined)
  assert.equal(parseSseRecord('data: not json'), undefined)
})

/** Run one stream of events through a translator. */
function translate(events: readonly Record<string, unknown>[]) {
  const translator = new StreamTranslator()
  return { translator, chunks: events.flatMap(event => translator.handle(event)) }
}

test('a text turn emits start, deltas, an authoritative block-end, usage, and finish', () => {
  const { translator, chunks } = translate([
    { type: 'message_start', message: { usage: { input_tokens: 11, output_tokens: 3 } } },
    { type: 'ping' },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '1' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '\n2' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 11, output_tokens: 9 } },
    { type: 'message_stop' },
  ])
  assert.equal(translator.terminal, true)
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: '1' },
    { type: 'text-delta', index: 0, text: '\n2' },
    { type: 'block-end', index: 0, block: { type: 'text', text: '1\n2' } },
    { type: 'usage', usage: { inputTokens: 11, outputTokens: 9, totalTokens: 20 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('a tool turn emits a raw-JSON argument stream and a complete tool-call block', () => {
  const { chunks } = translate([
    { type: 'message_start', message: { usage: { input_tokens: 5 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'toolu_1', name: 'read', input: {} } },
    { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"pa' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: 'th":"a"}' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 12 } },
    { type: 'message_stop' },
  ])
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'tool-call' },
    { type: 'tool-call-delta', index: 0, id: 'toolu_1', name: 'read', argumentsDelta: '' },
    { type: 'tool-call-delta', index: 0, id: 'toolu_1', argumentsDelta: '{"pa' },
    { type: 'tool-call-delta', index: 0, id: 'toolu_1', argumentsDelta: 'th":"a"}' },
    { type: 'block-end', index: 0, block: { type: 'tool-call', id: 'toolu_1', name: 'read', arguments: '{"path":"a"}' } },
    { type: 'usage', usage: { inputTokens: 5, outputTokens: 12, totalTokens: 17 } },
    { type: 'finish', reason: { kind: 'tool-calls' } },
  ])
})

test('thinking and server-tool blocks are declined without desynchronizing indexes', () => {
  const { chunks } = translate([
    { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'hmm' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'abc' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'hi' } },
    { type: 'content_block_stop', index: 1 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' } },
    { type: 'message_stop' },
  ])
  assert.deepEqual(chunks[0], { type: 'block-start', index: 1, blockType: 'text' })
  assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'stop' } })
  assert.equal(chunks.some(chunk => chunk.type === 'block-end' && chunk.index === 0), false)
})

test('a stream-level error event is a terminal failure chunk', () => {
  const { translator, chunks } = translate([
    { type: 'message_start', message: { usage: { input_tokens: 1 } } },
    { type: 'error', error: { type: 'overloaded_error', message: 'Overloaded' } },
  ])
  assert.equal(translator.terminal, true)
  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: { kind: 'error', failure: { message: 'google-vertex: Overloaded', code: 'SERVER' } },
  }])
})

test('usage is emitted only when the provider reported a counter', () => {
  const { chunks } = translate([
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hi' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' } },
    { type: 'message_stop' },
  ])
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'hi' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'hi' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])

  // An empty usage object is not a report either.
  const { chunks: empty } = translate([
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hi' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 'lots' } },
    { type: 'message_stop' },
  ])
  assert.equal(empty.some(chunk => chunk.type === 'usage'), false)
})

test('a completed turn with no content is an EMPTY_RESPONSE failure', () => {
  const { chunks } = translate([
    { type: 'message_start', message: { usage: { input_tokens: 1 } } },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 0 } },
    { type: 'message_stop' },
  ])
  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: {
      kind: 'error',
      failure: {
        message: 'google-vertex: the model completed the response with no content',
        code: 'EMPTY_RESPONSE',
      },
    },
  }])
})

test('events after a terminal event are ignored', () => {
  const { translator } = translate([
    { type: 'message_stop' },
  ])
  assert.deepEqual(translator.handle({ type: 'content_block_start', index: 0, content_block: { type: 'text' } }), [])
})
