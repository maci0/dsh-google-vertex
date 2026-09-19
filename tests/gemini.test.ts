/**
 * Pure Gemini wire tests: endpoint shape, request projection (including the
 * thought-signature replay Vertex requires), usage accounting, finish mapping,
 * and the stream translator.
 *
 * @module dsh-google-vertex/tests/gemini
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildGeminiRequest,
  DEFAULT_GEMINI_MAX_TOKENS,
  geminiEndpointFor,
  geminiReplayState,
  GeminiStreamTranslator,
  mapGeminiFinishReason,
  mapGeminiUsage,
  readGeminiReplay,
  SAFETY_BLOCKED_CODE,
  type GeminiContent,
  type GeminiRequestBody,
} from '../src/gemini.ts'
import type { VertexWireConfig } from '../src/wire.ts'
import type { Message } from '../src/host.ts'

const CONFIG: VertexWireConfig = { project: 'p1', location: 'global', maxTokens: DEFAULT_GEMINI_MAX_TOKENS }
const MODEL = 'gemini-3.5-flash'

/** The first turn of the assistant content in a built body. */
function modelParts(body: GeminiRequestBody, index = 1): GeminiContent | undefined {
  return body.contents[index]
}

test('the Gemini endpoint uses the global host and the SSE encoding', () => {
  assert.equal(
    geminiEndpointFor('p1', 'global', 'gemini-3.5-flash'),
    'https://aiplatform.googleapis.com/v1/projects/p1/locations/global/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse',
  )
  assert.equal(
    geminiEndpointFor('p1', 'us-east5', 'gemini-3.5-flash'),
    'https://us-east5-aiplatform.googleapis.com/v1/projects/p1/locations/us-east5/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse',
  )
})

test('buildGeminiRequest hoists system text, maps tools, and sends no thinking config', () => {
  const body = buildGeminiRequest({
    model: MODEL,
    system: 'one-shot system',
    temperature: 0.2,
    stop: ['STOP'],
    tools: [{ name: 'get_weather', description: 'weather', parameters: { type: 'object', properties: {} } }],
    messages: [
      { id: '0', role: 'system', content: [{ type: 'text', text: 'loop system' }] },
      { id: '1', role: 'user', content: [{ type: 'text', text: 'hi' }] },
      { id: '2', role: 'assistant', content: [{ type: 'reasoning', text: 'internal' }, { type: 'text', text: 'hello' }] },
      { id: '3', role: 'user', content: [{ type: 'text', text: 'more' }] },
    ],
  }, CONFIG)

  assert.equal(body.systemInstruction?.parts[0]?.text, 'one-shot system\n\nloop system')
  // The assistant's reasoning block is dropped; its text survives.
  assert.deepEqual(modelParts(body)?.parts, [{ text: 'hello' }])
  // Two consecutive user turns merge into one.
  assert.deepEqual(body.contents[2], { role: 'user', parts: [{ text: 'more' }] })
  assert.equal(body.tools?.[0]?.functionDeclarations[0]?.name, 'get_weather')
  assert.deepEqual(body.generationConfig, { maxOutputTokens: DEFAULT_GEMINI_MAX_TOKENS, temperature: 0.2, stopSequences: ['STOP'] })
  // Gemini 2.5 Pro refuses a zero thinking budget, so none is ever sent.
  assert.equal('thinkingConfig' in (body.generationConfig ?? {}), false)
})

test('a tool round trip carries the function name and Vertex thought signature', () => {
  const assistant: Message = {
    id: '2',
    role: 'assistant',
    content: [{ type: 'tool-call', id: 'call_74136', name: 'get_weather', arguments: '{"city":"Paris"}' }],
    source: { kind: 'model', replayState: geminiReplayState(MODEL, [{ type: 'tool-call', thoughtSignature: 'sig-abc' }]) },
  }
  const body = buildGeminiRequest({
    model: MODEL,
    messages: [
      { id: '1', role: 'user', content: [{ type: 'text', text: 'weather in Paris?' }] },
      assistant,
      { id: '3', role: 'user', content: [{ type: 'tool-result', toolCallId: 'call_74136', content: [{ type: 'text', text: '18C and clear' }] }] },
    ],
  }, CONFIG)

  assert.deepEqual(body.contents[1]?.parts, [{
    functionCall: { name: 'get_weather', args: { city: 'Paris' }, id: 'call_74136' },
    thoughtSignature: 'sig-abc',
  }])
  // The result names the function, not the call id: that is how Vertex matches them.
  assert.deepEqual(body.contents[2]?.parts, [{
    functionResponse: { name: 'get_weather', id: 'call_74136', response: { result: '18C and clear' } },
  }])
})

test('readGeminiReplay refuses a foreign, cross-model, or misaligned envelope', () => {
  const message: Message = {
    id: '2',
    role: 'assistant',
    content: [{ type: 'tool-call', id: 'c1', name: 't', arguments: '{}' }],
    source: { kind: 'model', replayState: geminiReplayState(MODEL, [{ type: 'tool-call', thoughtSignature: 'sig' }]) },
  }
  assert.deepEqual(readGeminiReplay(message, MODEL), [{ type: 'tool-call', thoughtSignature: 'sig' }])
  assert.equal(readGeminiReplay(message, 'gemini-2.5-pro'), undefined)
  assert.equal(readGeminiReplay({ ...message, source: { kind: 'plugin' } }, MODEL), undefined)
  assert.equal(readGeminiReplay({ ...message, source: { kind: 'model', replayState: { response: { kind: 'other' } } } }, MODEL), undefined)
  assert.equal(readGeminiReplay({
    ...message,
    source: { kind: 'model', replayState: geminiReplayState(MODEL, []) },
  }, MODEL), undefined)
})

test('mapGeminiUsage keeps cached input disjoint and folds thinking into output', () => {
  assert.deepEqual(mapGeminiUsage({
    promptTokenCount: 1000,
    cachedContentTokenCount: 400,
    candidatesTokenCount: 20,
    thoughtsTokenCount: 53,
    totalTokenCount: 1073,
  }), {
    inputTokens: 600,
    outputTokens: 73,
    totalTokens: 1073,
    cacheReadTokens: 400,
    reasoningTokens: 53,
  })
  assert.deepEqual(mapGeminiUsage({ promptTokenCount: 5, candidatesTokenCount: 2 }), {
    inputTokens: 5,
    outputTokens: 2,
    totalTokens: 7,
  })
  // An exact total is the provider's; a partial report omits it rather than
  // presenting a one-sided sum as the whole request.
  assert.deepEqual(mapGeminiUsage({ promptTokenCount: 5 }), { inputTokens: 5, outputTokens: 0 })
  assert.deepEqual(mapGeminiUsage({ candidatesTokenCount: 2 }), { inputTokens: 0, outputTokens: 2 })
  assert.deepEqual(mapGeminiUsage({ totalTokenCount: 9 }), { inputTokens: 0, outputTokens: 0, totalTokens: 9 })
})

test('mapGeminiFinishReason covers tools, truncation, and safety refusals', () => {
  assert.deepEqual(mapGeminiFinishReason('STOP', false), { kind: 'stop' })
  // Gemini reports a function call as an ordinary STOP, so the adapter's own
  // observation of a tool block is what makes it a tool-call finish.
  assert.deepEqual(mapGeminiFinishReason('STOP', true), { kind: 'tool-calls' })
  assert.deepEqual(mapGeminiFinishReason('MAX_TOKENS', false), { kind: 'max-tokens' })
  const safety = mapGeminiFinishReason('PROHIBITED_CONTENT', false)
  assert.equal(safety.kind, 'error')
  assert.equal(safety.kind === 'error' ? safety.failure.code : '', SAFETY_BLOCKED_CODE)
  assert.equal(mapGeminiFinishReason('MALFORMED_FUNCTION_CALL', false).kind, 'error')
})

test('the translator joins split text parts into one block and closes it once', () => {
  const translator = new GeminiStreamTranslator(MODEL)
  const emitted = [
    ...translator.handle({ candidates: [{ content: { role: 'model', parts: [{ text: 'Hello ' }] } }] }),
    ...translator.handle({ candidates: [{ content: { role: 'model', parts: [{ text: 'world' }] } }] }),
    ...translator.handle({
      candidates: [{ content: { role: 'model', parts: [{ text: '' }] }, finishReason: 'STOP' }],
      usageMetadata: { promptTokenCount: 9, candidatesTokenCount: 3, totalTokenCount: 12 },
    }),
  ]
  assert.equal(translator.sawFinish, true)
  assert.deepEqual(emitted, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hello ' },
    { type: 'text-delta', index: 0, text: 'world' },
  ])
  assert.deepEqual(translator.finish(), [
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello world' } },
    { type: 'usage', usage: { inputTokens: 9, outputTokens: 3, totalTokens: 12 } },
    {
      type: 'finish',
      reason: { kind: 'stop' },
      replayState: geminiReplayState(MODEL, [{ type: 'text' }]),
    },
  ])
})

test('a function call becomes a complete tool-call block with its signature recorded', () => {
  const translator = new GeminiStreamTranslator(MODEL)
  const emitted = translator.handle({
    candidates: [{
      content: {
        role: 'model',
        parts: [
          { text: 'Checking. ' },
          { functionCall: { name: 'get_weather', args: { city: 'Paris' }, id: 'call_1' }, thoughtSignature: 'sig-1' },
        ],
      },
      finishReason: 'STOP',
    }],
    usageMetadata: { promptTokenCount: 34, candidatesTokenCount: 16, thoughtsTokenCount: 53, totalTokenCount: 103 },
  })

  assert.deepEqual(emitted, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Checking. ' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Checking. ' } },
    { type: 'block-start', index: 1, blockType: 'tool-call' },
    { type: 'tool-call-delta', index: 1, id: 'call_1', name: 'get_weather', argumentsDelta: '{"city":"Paris"}' },
    { type: 'block-end', index: 1, block: { type: 'tool-call', id: 'call_1', name: 'get_weather', arguments: '{"city":"Paris"}' } },
  ])

  const terminal = translator.finish()
  assert.deepEqual(terminal.at(-2), {
    type: 'usage',
    usage: { inputTokens: 34, outputTokens: 69, totalTokens: 103, reasoningTokens: 53 },
  })
  assert.deepEqual(terminal.at(-1), {
    type: 'finish',
    reason: { kind: 'tool-calls' },
    replayState: geminiReplayState(MODEL, [{ type: 'text' }, { type: 'tool-call', thoughtSignature: 'sig-1' }]),
  })
})

test('an in-band error envelope ends the stream with the provider classification', () => {
  const translator = new GeminiStreamTranslator(MODEL)
  const chunks = translator.handle({ error: { code: 429, message: 'Rate limit exceeded', status: 'RESOURCE_EXHAUSTED' } })

  assert.equal(translator.failed, true)
  assert.equal(translator.sawFinish, false)
  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: { kind: 'error', failure: { message: 'google-vertex: Rate limit exceeded', code: 'QUOTA' } },
  }])
  // Terminal: a later payload cannot add a second terminal chunk.
  assert.deepEqual(translator.handle({ candidates: [{ content: { parts: [{ text: 'late' }] }, finishReason: 'STOP' }] }), [])
})

test('usage is emitted only when the provider reported a counter', () => {
  const translator = new GeminiStreamTranslator(MODEL)
  translator.handle({
    candidates: [{ content: { parts: [{ text: 'Hi' }] }, finishReason: 'STOP' }],
  })
  assert.deepEqual(translator.finish(), [
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hi' } },
    {
      type: 'finish',
      reason: { kind: 'stop' },
      replayState: geminiReplayState(MODEL, [{ type: 'text' }]),
    },
  ])
})

test('thought summaries are dropped and a contentless response is an empty-response failure', () => {
  const translator = new GeminiStreamTranslator(MODEL)
  const emitted = translator.handle({
    candidates: [{
      content: { role: 'model', parts: [{ thought: true, text: 'thinking hard' }] },
      finishReason: 'STOP',
    }],
  })
  assert.deepEqual(emitted, [])
  assert.deepEqual(translator.finish(), [{
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
