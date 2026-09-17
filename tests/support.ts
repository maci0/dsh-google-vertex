/**
 * Shared adapter-test fixtures: a stubbed streaming response, a counting token
 * source, a body that stalls until its request is aborted, and the collector
 * both adapter suites feed their streams through.
 *
 * @module dsh-google-vertex/tests/support
 */

import type { GenerateOptions, StreamChunk } from '../src/host.ts'

/** A streaming response whose body arrives in the given chunks. */
export function streamResponse(chunks: readonly string[], init: ResponseInit = {}): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  }), { status: 200, headers: { 'content-type': 'text/event-stream' }, ...init })
}

/** A fixed token source that counts how often it was asked. */
export function tokens(value = 'access-token'): { get(): Promise<string>; calls: () => number } {
  let calls = 0
  return {
    async get() {
      calls += 1
      return value
    },
    calls: () => calls,
  }
}

/**
 * A body that sends the given chunks and then stalls until the request signal
 * aborts — which is how a real `fetch` body answers an aborted read.
 */
export function stallingResponse(signal: AbortSignal | null | undefined, before: readonly string[] = []): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of before) controller.enqueue(encoder.encode(chunk))
      signal?.addEventListener('abort', () => controller.error(new Error('The operation was aborted')), { once: true })
    },
  }), { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

/** Collect one adapter stream. */
export async function collect(
  adapter: { stream(options: GenerateOptions): AsyncIterable<StreamChunk> },
  options: GenerateOptions,
): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of adapter.stream(options)) chunks.push(chunk)
  return chunks
}
