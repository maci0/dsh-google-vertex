/**
 * Tests for the model discovery module: the Vertex catalog fetch, the TTL
 * cache, and the fallback-on-failure behavior.
 *
 * @module dsh-google-vertex/tests/discovery
 */

import assert from 'node:assert/strict'
import { mock, test } from 'node:test'

import { DEFAULT_CACHE_TTL_MS, fetchGeminiModels, ModelCache, probeAnthropicModels } from '../src/discovery.ts'
import type { FetchLike } from '../src/auth.ts'

// ---------------------------------------------------------------------------
// ModelCache
// ---------------------------------------------------------------------------

test('ModelCache returns fallback when no fetch function is provided', async () => {
  const fallback = [{ id: 'a', name: 'A' }]
  const cache = new ModelCache(fallback)
  const result = await cache.get()
  assert.deepEqual(result, fallback)
})

test('ModelCache returns fetched models and caches them', async () => {
  const fallback = [{ id: 'fallback', name: 'Fallback' }]
  const fetched = [{ id: 'live', name: 'Live' }]
  let calls = 0
  const cache = new ModelCache(fallback)
  const fetchFn = async () => { calls += 1; return fetched }

  const first = await cache.get(fetchFn)
  assert.deepEqual(first, fetched)
  assert.equal(calls, 1)

  // Second call returns cache, no new fetch.
  const second = await cache.get(fetchFn)
  assert.deepEqual(second, fetched)
  assert.equal(calls, 1)
})

test('ModelCache falls back when fetch throws', async () => {
  const fallback = [{ id: 'safe', name: 'Safe' }]
  const cache = new ModelCache(fallback)
  const fetchFn = async () => { throw new Error('network') }

  const result = await cache.get(fetchFn)
  assert.deepEqual(result, fallback)
})

test('ModelCache falls back when fetch returns empty', async () => {
  const fallback = [{ id: 'default', name: 'Default' }]
  const cache = new ModelCache(fallback)
  const fetchFn = async (): Promise<readonly { id: string; name: string }[]> => []

  const result = await cache.get(fetchFn)
  assert.deepEqual(result, fallback)
})

test('ModelCache invalidate forces re-fetch', async () => {
  const fallback = [{ id: 'old', name: 'Old' }]
  let generation = 0
  const cache = new ModelCache(fallback)
  const fetchFn = async () => {
    generation += 1
    return [{ id: `model-${generation}`, name: `Model ${generation}` }]
  }

  await cache.get(fetchFn)
  cache.invalidate()
  const result = await cache.get(fetchFn)
  assert.equal(result[0]?.id, 'model-2')
})

test('ModelCache shares concurrent in-flight fetches', async () => {
  const fallback = [{ id: 'x', name: 'X' }]
  let calls = 0
  const cache = new ModelCache(fallback)
  const fetchFn = async () => {
    calls += 1
    await new Promise(resolve => setTimeout(resolve, 10))
    return [{ id: 'shared', name: 'Shared' }]
  }

  const [a, b] = await Promise.all([cache.get(fetchFn), cache.get(fetchFn)])
  assert.deepEqual(a, b)
  assert.equal(calls, 1)
})

test('ModelCache respects TTL expiry', async () => {
  const fallback = [{ id: 'x', name: 'X' }]
  let calls = 0
  const cache = new ModelCache(fallback)
  const fetchFn = async () => {
    calls += 1
    return [{ id: `v${calls}`, name: `V${calls}` }]
  }

  await cache.get(fetchFn)
  assert.equal(calls, 1)

  // Advance the clock past the five-minute TTL. The mocked clock starts at the
  // epoch, so anchor it to the real time of the first fetch first.
  const start = Date.now()
  mock.timers.enable({ apis: ['Date'] })
  try {
    mock.timers.setTime(start + DEFAULT_CACHE_TTL_MS + 1)
    await cache.get(fetchFn)
  } finally {
    mock.timers.reset()
  }
  assert.equal(calls, 2)
})

// ---------------------------------------------------------------------------
// fetchGeminiModels
// ---------------------------------------------------------------------------

test('fetchGeminiModels parses a single-page response', async () => {
  const fetch: FetchLike = (_url, _init) => Promise.resolve(new Response(
    JSON.stringify({
      models: [
        { name: 'publishers/google/models/gemini-2.5-pro', displayName: 'Gemini 2.5 Pro' },
        { name: 'publishers/google/models/gemini-2.5-flash', displayName: 'Gemini 2.5 Flash' },
      ],
    }),
    { status: 200, headers: { 'content-type': 'application/json' } },
  ))
  const tokenSource = { get: async () => 'token' }

  const models = await fetchGeminiModels('global', tokenSource, fetch)
  assert.equal(models.length, 2)
  assert.equal(models[0]?.id, 'gemini-2.5-pro')
  assert.equal(models[0]?.name, 'Gemini 2.5 Pro (Vertex)')
  assert.equal(models[1]?.id, 'gemini-2.5-flash')
})

test('fetchGeminiModels follows pagination', async () => {
  let page = 0
  const fetch: FetchLike = (url, _init) => {
    page += 1
    if (page === 1) {
      assert.ok(!url.includes('pageToken'))
      return Promise.resolve(new Response(
        JSON.stringify({
          models: [{ name: 'publishers/google/models/gemini-a' }],
          nextPageToken: 'tok2',
        }),
        { status: 200 },
      ))
    }
    assert.ok(url.includes('pageToken=tok2'))
    return Promise.resolve(new Response(
      JSON.stringify({ models: [{ name: 'publishers/google/models/gemini-b' }] }),
      { status: 200 },
    ))
  }
  const tokenSource = { get: async () => 'token' }

  const models = await fetchGeminiModels('us-central1', tokenSource, fetch)
  assert.equal(models.length, 2)
  assert.equal(models[0]?.id, 'gemini-a')
  assert.equal(models[1]?.id, 'gemini-b')
})

test('fetchGeminiModels skips models without generateContent support', async () => {
  const fetch: FetchLike = (_url, _init) => Promise.resolve(new Response(
    JSON.stringify({
      models: [
        {
          name: 'publishers/google/models/gemini-pro',
          supportedActions: ['generateContent', 'streamGenerateContent'],
        },
        {
          name: 'publishers/google/models/text-embedding-004',
          supportedActions: ['embedContent'],
        },
      ],
    }),
    { status: 200 },
  ))
  const tokenSource = { get: async () => 'token' }

  const models = await fetchGeminiModels('global', tokenSource, fetch)
  assert.equal(models.length, 1)
  assert.equal(models[0]?.id, 'gemini-pro')
})

test('fetchGeminiModels throws on HTTP error', async () => {
  const fetch: FetchLike = (_url, _init) => Promise.resolve(new Response('', { status: 403 }))
  const tokenSource = { get: async () => 'token' }

  await assert.rejects(
    () => fetchGeminiModels('global', tokenSource, fetch),
    /HTTP 403/,
  )
})

test('fetchGeminiModels uses correct endpoint for regional location', async () => {
  let calledUrl = ''
  const fetch: FetchLike = (url, _init) => {
    calledUrl = url
    return Promise.resolve(new Response(JSON.stringify({ models: [] }), { status: 200 }))
  }
  const tokenSource = { get: async () => 'token' }

  await fetchGeminiModels('us-east4', tokenSource, fetch)
  assert.ok(calledUrl.startsWith('https://us-east4-aiplatform.googleapis.com/'))
  assert.ok(calledUrl.includes('/publishers/google/models'))
})

test('fetchGeminiModels uses global endpoint when location is global', async () => {
  let calledUrl = ''
  const fetch: FetchLike = (url, _init) => {
    calledUrl = url
    return Promise.resolve(new Response(JSON.stringify({ models: [] }), { status: 200 }))
  }
  const tokenSource = { get: async () => 'token' }

  await fetchGeminiModels('global', tokenSource, fetch)
  assert.ok(calledUrl.startsWith('https://aiplatform.googleapis.com/'))
})

// ---------------------------------------------------------------------------
// probeAnthropicModels
// ---------------------------------------------------------------------------

const CANDIDATES = [
  { id: 'claude-opus-4-6', name: 'Claude Opus 4.6 (Vertex)' },
  { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5 (Vertex)' },
  { id: 'claude-gone', name: 'Claude Gone (Vertex)' },
]

test('probeAnthropicModels keeps models that return 400 and drops 404', async () => {
  const fetch: FetchLike = (url, _init) => {
    const status = url.includes('claude-gone') ? 404 : 400
    return Promise.resolve(new Response('', { status }))
  }
  const tokenSource = { get: async () => 'token' }

  const result = await probeAnthropicModels(CANDIDATES, 'proj', 'global', tokenSource, fetch)
  assert.equal(result.length, 2)
  assert.equal(result[0]?.id, 'claude-opus-4-6')
  assert.equal(result[1]?.id, 'claude-sonnet-4-5')
})

test('probeAnthropicModels keeps models that return 200', async () => {
  const fetch: FetchLike = (_url, _init) =>
    Promise.resolve(new Response('', { status: 200 }))
  const tokenSource = { get: async () => 'token' }

  const result = await probeAnthropicModels(CANDIDATES, 'proj', 'us-east4', tokenSource, fetch)
  assert.equal(result.length, 3)
})

test('probeAnthropicModels keeps models on network error', async () => {
  const fetch: FetchLike = (_url, _init) => Promise.reject(new Error('network'))
  const tokenSource = { get: async () => 'token' }

  const result = await probeAnthropicModels(CANDIDATES, 'proj', 'global', tokenSource, fetch)
  assert.equal(result.length, 3)
})

test('probeAnthropicModels sends correct endpoint and body', async () => {
  const requests: { url: string; body: string }[] = []
  const fetch: FetchLike = async (url, init) => {
    requests.push({ url, body: typeof init.body === 'string' ? init.body : '' })
    return new Response('', { status: 400 })
  }
  const tokenSource = { get: async () => 'bearer-tok' }

  await probeAnthropicModels(
    [{ id: 'claude-sonnet-4-5', name: 'test' }],
    'my-project', 'us-central1', tokenSource, fetch,
  )
  assert.equal(requests.length, 1)
  const req = requests[0]!
  assert.ok(req.url.includes('us-central1-aiplatform.googleapis.com'))
  assert.ok(req.url.includes('/projects/my-project/'))
  assert.ok(req.url.includes('/publishers/anthropic/models/claude-sonnet-4-5:rawPredict'))
  const body = JSON.parse(req.body)
  assert.equal(body.anthropic_version, 'vertex-2023-10-16')
  assert.equal(body.max_tokens, 1)
  assert.deepEqual(body.messages, [])
})

test('probeAnthropicModels returns empty candidates unchanged', async () => {
  const fetch: FetchLike = () => { throw new Error('should not be called') }
  const tokenSource = { get: async () => 'token' }

  const result = await probeAnthropicModels([], 'proj', 'global', tokenSource, fetch)
  assert.equal(result.length, 0)
})

test('probeAnthropicModels keeps models on 429 and 500', async () => {
  let call = 0
  const fetch: FetchLike = (_url, _init) => {
    call++
    const status = call === 1 ? 429 : 500
    return Promise.resolve(new Response('', { status }))
  }
  const tokenSource = { get: async () => 'token' }

  const result = await probeAnthropicModels(
    [{ id: 'a', name: 'A' }, { id: 'b', name: 'B' }],
    'proj', 'global', tokenSource, fetch,
  )
  assert.equal(result.length, 2)
})
