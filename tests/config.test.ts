/**
 * Configuration tests: the credential source, the project/location defaults,
 * and the loud refusals a typo must produce at mount.
 *
 * @module dsh-google-vertex/tests/config
 */

import assert from 'node:assert/strict'
import { generateKeyPairSync } from 'node:crypto'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { apply, Config, DEFAULT_MODELS, GEMINI_PROVIDER, PROVIDER, resolveConfig } from '../src/index.ts'
import { DEFAULT_GEMINI_CONTEXT_WINDOW } from '../src/gemini.ts'
import type { HostContext } from '../src/host.ts'
import { DEFAULT_STREAM_IDLE_TIMEOUT_MS, MAX_TIMER_DELAY_MS } from '../src/wire.ts'

const { privateKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
  publicKeyEncoding: { type: 'spki', format: 'pem' },
})

const dir = mkdtempSync(join(tmpdir(), 'dsh-google-vertex-'))
const accountPath = join(dir, 'service-account.json')
writeFileSync(accountPath, JSON.stringify({
  type: 'service_account',
  project_id: 'example-project',
  client_email: 'sa@example.iam.gserviceaccount.com',
  private_key: privateKey,
  token_uri: 'https://oauth2.googleapis.com/token',
}))

const NO_ENV: NodeJS.ProcessEnv = {}

test('the row schema fills the code-side defaults and leaves absent keys absent', () => {
  const filled = Config({ serviceAccountFile: accountPath, project: 'p' })
  assert.equal(filled.location, 'global')
  assert.equal(filled.contextWindow, 200_000)
  assert.equal(filled.maxTokens, 32_000)
  assert.equal(filled.streamIdleTimeoutMs, DEFAULT_STREAM_IDLE_TIMEOUT_MS)
  // Neither catalog has a code-side default, so an omitted one materializes
  // empty and `resolveConfig`'s `length === 0` branch keeps the built-in list.
  assert.deepEqual(filled.models, [])
  assert.deepEqual(filled.geminiModels, [])
  assert.deepEqual(resolveConfig(filled, NO_ENV).anthropic.models, DEFAULT_MODELS)
  assert.throws(() => Config({ contextWindow: 0 }), /contextWindow/)
  assert.throws(() => Config({ streamIdleTimeoutMs: MAX_TIMER_DELAY_MS + 1 }), /streamIdleTimeoutMs/)
})

test('the credentials file supplies the project and the global default location', () => {
  const resolved = resolveConfig({ serviceAccountFile: accountPath }, NO_ENV)
  assert.equal(resolved.anthropic.project, 'example-project')
  assert.equal(resolved.anthropic.location, 'global')
  assert.deepEqual(resolved.anthropic.models, DEFAULT_MODELS)
  assert.equal(resolved.anthropic.maxTokens, 32_000)
  assert.equal(resolved.anthropic.contextWindow, 200_000)
})

test('an explicit project and region override the file and the default', () => {
  const resolved = resolveConfig(
    { serviceAccountFile: accountPath, project: 'other', location: 'us-east5', models: ['claude-opus-4-6'] },
    NO_ENV,
  )
  assert.equal(resolved.anthropic.project, 'other')
  assert.equal(resolved.anthropic.location, 'us-east5')
  assert.deepEqual(resolved.anthropic.models, [{ id: 'claude-opus-4-6', name: 'claude-opus-4-6' }])
})

test('the launch environment is the fallback credential and region source', () => {
  const resolved = resolveConfig({}, {
    GOOGLE_APPLICATION_CREDENTIALS: accountPath,
    GOOGLE_CLOUD_LOCATION: 'europe-west1',
  })
  assert.equal(resolved.anthropic.location, 'europe-west1')
})

test('an explicit Gemini catalog keeps the built-in wording for known ids', () => {
  const resolved = resolveConfig({ serviceAccountFile: accountPath, geminiModels: ['gemini-3.5-flash', 'gemini-9-future'] }, NO_ENV)
  assert.deepEqual(resolved.gemini.models, [
    { id: 'gemini-3.5-flash', name: 'Gemini 3.5 Flash (Vertex)' },
    { id: 'gemini-9-future', name: 'gemini-9-future' },
  ])
  // Capacities are not per model: every id serves Vertex's exclusive output
  // ceiling minus one, and the family context window.
  assert.equal(resolved.gemini.maxTokens, 65_535)
  assert.equal(resolved.gemini.models[1]?.id, 'gemini-9-future')
  assert.equal(DEFAULT_GEMINI_CONTEXT_WINDOW, 1_048_576)
})

test('the project falls back to the launch environment before the file', () => {
  const barePath = join(dir, 'no-project.json')
  writeFileSync(barePath, JSON.stringify({
    client_email: 'sa@example.iam.gserviceaccount.com',
    private_key: privateKey,
  }))
  const fromEnv = resolveConfig({ serviceAccountFile: barePath }, { GOOGLE_CLOUD_PROJECT: 'env-project' })
  assert.equal(fromEnv.anthropic.project, 'env-project')
  assert.equal(
    resolveConfig({ serviceAccountFile: barePath }, { GCLOUD_PROJECT: 'legacy-project' }).anthropic.project,
    'legacy-project',
  )
  assert.throws(() => resolveConfig({ serviceAccountFile: barePath }, NO_ENV), /no project configured/)
})

test('a missing credential, a missing project, and a bad region each fail loudly', () => {
  assert.throws(() => resolveConfig({}, NO_ENV), /no service account configured/)
  assert.throws(
    () => resolveConfig({ serviceAccountFile: join(dir, 'absent.json') }, NO_ENV),
    /cannot read service account file/,
  )
  assert.throws(
    () => resolveConfig({ serviceAccountFile: accountPath, location: 'US East' }, NO_ENV),
    /not a valid Vertex region/,
  )
  assert.throws(
    () => resolveConfig({ serviceAccountFile: accountPath, contextWindow: 0 }, NO_ENV),
    /contextWindow must be a positive integer/,
  )
})

test('the idle bound reaches both routes and refuses a value no timer can hold', () => {
  const resolved = resolveConfig({ serviceAccountFile: accountPath, streamIdleTimeoutMs: 1_500 }, NO_ENV)
  assert.equal(resolved.anthropic.streamIdleTimeoutMs, 1_500)
  assert.equal(resolved.gemini.streamIdleTimeoutMs, 1_500)

  const defaulted = resolveConfig({ serviceAccountFile: accountPath }, NO_ENV)
  assert.equal(defaulted.anthropic.streamIdleTimeoutMs, DEFAULT_STREAM_IDLE_TIMEOUT_MS)
  assert.equal(defaulted.gemini.streamIdleTimeoutMs, DEFAULT_STREAM_IDLE_TIMEOUT_MS)

  for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, MAX_TIMER_DELAY_MS + 1]) {
    assert.throws(
      () => resolveConfig({ serviceAccountFile: accountPath, streamIdleTimeoutMs: bad }, NO_ENV),
      /streamIdleTimeoutMs must be a positive finite number/,
    )
  }
})

test('apply registers both routes and logs what it read', () => {
  const registered: string[][] = []
  const logs: unknown[] = []
  const ctx = {
    llm: { registerAdapter: (providers: string[]) => { registered.push(providers); return () => {} } },
    logger: { info: (message: unknown) => { logs.push(message) }, warn: () => {} },
  } as unknown as HostContext

  apply(ctx, { serviceAccountFile: accountPath })
  assert.deepEqual(registered, [[PROVIDER], [GEMINI_PROVIDER]])
  assert.match(String(logs[0]), /google-vertex-anthropic/)
  assert.match(String(logs[0]), /google-vertex-gemini/)
  assert.match(String(logs[0]), /example-project/)
})
