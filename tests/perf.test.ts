/**
 * Deterministic performance test for the per-chunk streaming path.
 *
 * Wall clock is not asserted: it moves with frequency scaling, turbo, and noisy
 * neighbours. The gate is `process.cpuUsage()` — CPU time actually consumed by
 * this process, which descheduling and I/O wait do not inflate — measured as the
 * median of several runs over a fixed, network-free chunk stream, with the first
 * run dropped. A fixed workload shape keeps the number comparable across hosts.
 *
 * The band is deliberately wide (4x the recorded baseline): it is a loaded-CI
 * gate for an algorithmic regression — a per-record regex, an O(n²) rebuild of
 * the record buffer, a re-parse of the whole stream per chunk — not a
 * micro-benchmark. Tighten it on dedicated hardware, never below 2x.
 *
 * @module dsh-google-vertex/tests/perf
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { GoogleVertexAnthropicAdapter } from '../src/adapter.ts'
import { parseSseRecord, SseRecordReader, StreamTranslator } from '../src/wire.ts'

/** Records in the fixed stream; large enough that CPU time dwarfs timer noise. */
const RECORDS = 20_000

/**
 * Median CPU milliseconds this workload cost on the recorded host, and the
 * tolerance multiplier. Baseline: AMD Ryzen 9 9950X, node v26.9.0, pinned to
 * one core (`taskset -c 2`). The runner itself roughly doubles this number, so
 * the constant is the median observed under `node --test`, not under a bare
 * script.
 */
const BASELINE_CPU_MS = 45
const TOLERANCE = 4

/** A deterministic byte source, so every run replays the identical stream. */
function pseudoRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0x1_0000_0000
  }
}

/**
 * The fixed Anthropic publisher stream: a message start, one text block whose
 * deltas arrive as `RECORDS` records, and the events that close the turn.
 */
function fixedStream(): string {
  const random = pseudoRandom(0x5eed)
  const words = ['streaming', 'tokens', 'arrive', 'here', 'and', 'the', 'buffer', 'frames']
  const parts = [
    'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":1204,"output_tokens":1}}}\n\n',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
  ]
  for (let index = 0; index < RECORDS; index += 1) {
    let text = ''
    for (let word = 0; word < 1 + Math.floor(random() * 4); word += 1) {
      text += `${words[Math.floor(random() * words.length)]} `
    }
    parts.push(`event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"${text}"}}\n\n`)
  }
  parts.push(
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4096}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
  )
  return parts.join('')
}

/** Split the stream into transport chunks at fixed, uneven boundaries. */
function transportChunks(text: string): Uint8Array[] {
  const random = pseudoRandom(0xa11ce)
  const encoder = new TextEncoder()
  const chunks: Uint8Array[] = []
  for (let offset = 0; offset < text.length;) {
    const size = 96 + Math.floor(random() * 3000)
    chunks.push(encoder.encode(text.slice(offset, offset + size)))
    offset += size
  }
  return chunks
}

/** Drive the real parse and translate functions over the fixed chunk stream. */
async function run(chunks: readonly Uint8Array[]): Promise<number> {
  const translator = new StreamTranslator()
  let emitted = 0
  const body = (async function* () { for (const chunk of chunks) yield chunk })()
  const reader = new SseRecordReader(body, { armIdle() {}, clearIdle() {} })
  try {
    for (let records = await reader.read(); records !== undefined; records = await reader.read()) {
      for (const record of records) {
        const event = parseSseRecord(record)
        if (event === undefined) continue
        for (const _chunk of translator.handle(event)) emitted += 1
      }
    }
  } finally {
    await reader.close()
  }
  return emitted
}

test('the streaming path stays within its CPU-time band', async () => {
  const chunks = transportChunks(fixedStream())
  // Workload sanity: a stream that framed nothing would pass any band, so the
  // chunk count the translator must produce is asserted before the timing.
  assert.equal(await run(chunks), RECORDS + 4)

  const samples: number[] = []
  for (let iteration = 0; iteration < 6; iteration += 1) {
    const before = process.cpuUsage()
    await run(chunks)
    const used = process.cpuUsage(before)
    samples.push((used.user + used.system) / 1000)
  }
  samples.shift() // drop the first: JIT, inline caches, and cold string shapes
  const sorted = [...samples].sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0
  console.log(`# streaming path CPU median ${median.toFixed(1)}ms over ${RECORDS} records (band ${BASELINE_CPU_MS * TOLERANCE}ms)`)

  assert.ok(
    median < BASELINE_CPU_MS * TOLERANCE,
    `streaming path CPU median ${median.toFixed(1)}ms exceeds ${BASELINE_CPU_MS * TOLERANCE}ms`
    + ` (${TOLERANCE}x recorded baseline ${BASELINE_CPU_MS}ms over ${RECORDS} records)`,
  )
})

/** The adapter's own streaming pipeline over a stubbed transport. */
async function runAdapter(chunks: readonly Uint8Array[]): Promise<number> {
  const adapter = new GoogleVertexAnthropicAdapter({
    serviceAccount: { client_email: 'perf@example.com', private_key: 'unused' },
    project: 'p1',
    location: 'global',
    models: [{ id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5 (Vertex)' }],
    contextWindow: 200_000,
    maxTokens: 32_000,
    streamIdleTimeoutMs: 300_000,
  }, {
    fetch: () => Promise.resolve(new Response(new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk)
        controller.close()
      },
    }), { status: 200, headers: { 'content-type': 'text/event-stream' } })),
    tokens: { get: () => Promise.resolve('access-token') },
  })
  let emitted = 0
  for await (const _chunk of adapter.stream({
    model: 'claude-sonnet-4-5',
    messages: [{ id: '1', role: 'user', content: [{ type: 'text', text: 'hi' }] }],
  })) emitted += 1
  return emitted
}

test('the idle watchdog arms a bounded number of timers per stream', async () => {
  const chunks = transportChunks(fixedStream())
  const realSetTimeout = globalThis.setTimeout
  const realClearTimeout = globalThis.clearTimeout
  let armed = 0
  globalThis.setTimeout = ((callback: () => void, ms?: number) => {
    armed += 1
    return realSetTimeout(callback, ms)
  }) as typeof globalThis.setTimeout
  try {
    const emitted = await runAdapter(chunks)
    // Workload sanity first: a stream that framed nothing would arm nothing.
    assert.equal(emitted, RECORDS + 4)
    // One timer covers the whole stream. Anything proportional to
    // `chunks.length` is the per-read `setTimeout`/`clearTimeout` pair this
    // bounds — the shape that costs more the faster the provider streams.
    assert.ok(armed <= 4, `watchdog armed ${armed} timers over ${chunks.length} transport reads`)
  } finally {
    globalThis.setTimeout = realSetTimeout
    globalThis.clearTimeout = realClearTimeout
  }
})
