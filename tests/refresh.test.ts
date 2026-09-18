/**
 * Refresh test: the manual re-discovery the browser half's card asks for.
 *
 * The card can only reach the host process through a settings write, so the
 * whole path is: settings commit → the plugin's change hook → both adapters
 * drop their cached catalogs → the next catalog read reaches Vertex again. The
 * test drives that path over a real Cordis `Context`, a stub `settings` service
 * that records the hook, and a stubbed transport that counts catalog requests.
 *
 * @module dsh-google-vertex/tests/refresh
 */

import assert from 'node:assert/strict'
import { generateKeyPairSync } from 'node:crypto'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { Context, Service } from '@deepseek-ai/cordis'
import type { Plugin } from '@deepseek-ai/cordis'

import * as plugin from '../src/index.ts'
import type { LlmAdapterLike, SettingsSectionHooksLike } from '../src/host.ts'

/** The narrowest stand-in for `ctx.llm`, owning routes through the calling fiber. */
class StubLlm extends Service {
  /** Provider route → adapter, as the service holds them. */
  readonly routes = new Map<string, unknown>()

  constructor(ctx: Context) {
    super(ctx, 'llm')
  }

  registerAdapter(providers: readonly string[], adapter: unknown): () => void {
    const dispose = this.ctx.effect(() => {
      for (const provider of providers) this.routes.set(provider, adapter)
      return () => {
        for (const provider of providers) this.routes.delete(provider)
      }
    })
    return () => void dispose()
  }
}

/** A `ctx.settings` that records the section the plugin installs, and nothing else. */
class StubSettings extends Service {
  /** One entry per namespace the plugin registered. */
  readonly installs: { namespace: string; hooks: SettingsSectionHooksLike }[] = []

  constructor(ctx: Context) {
    super(ctx, 'settings')
  }

  installSection(
    _owner: unknown,
    namespace: string,
    _schema: unknown,
    _entry: unknown,
    hooks: SettingsSectionHooksLike,
  ): void {
    this.installs.push({ namespace, hooks })
  }
}

/** An adapter with the refresh verb the plugin's settings hook calls. */
type RefreshableAdapter = LlmAdapterLike & { invalidateModels(): void }

test('a committed settings write drops both cached catalogs, so the next read re-discovers', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-google-vertex-refresh-'))
  const accountPath = join(dir, 'service-account.json')
  // A real key: the token source signs the assertion before it ever reaches the
  // stubbed transport, and a placeholder key would fail there instead.
  const { privateKey } = generateKeyPairSync('rsa', {
    modulusLength: 2048,
    publicKeyEncoding: { type: 'spki', format: 'pem' },
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
  })
  writeFileSync(accountPath, JSON.stringify({
    type: 'service_account',
    project_id: 'refresh-project',
    client_email: 'sa@example.iam.gserviceaccount.com',
    private_key: privateKey,
    token_uri: 'https://oauth2.googleapis.com/token',
  }))

  /** Catalog requests, the only traffic that matters here. */
  const catalog = { gemini: 0, anthropic: 0 }
  const realFetch = globalThis.fetch
  globalThis.fetch = ((input: string | URL | Request): Promise<Response> => {
    const url = String(input)
    if (url.includes('oauth2.googleapis.com')) {
      return Promise.resolve(new Response(JSON.stringify({ access_token: 'token', expires_in: 3600 }), { status: 200 }))
    }
    if (url.includes('/publishers/google/models')) {
      catalog.gemini += 1
      return Promise.resolve(new Response(JSON.stringify({
        models: [{ name: 'publishers/google/models/gemini-9-live', displayName: 'Gemini 9 Live' }],
      }), { status: 200 }))
    }
    // Anthropic discovery probes one model id per request; anything but 404 keeps it.
    catalog.anthropic += 1
    return Promise.resolve(new Response('{}', { status: 400 }))
  }) as typeof globalThis.fetch

  try {
    const ctx = new Context()
    const llm = new StubLlm(ctx)
    const settings = new StubSettings(ctx)

    const fiber = await ctx.plugin(plugin as unknown as Plugin, { serviceAccountFile: accountPath })

    // The browser card is keyed on this namespace; the host half must serve it.
    assert.deepEqual(settings.installs.map(entry => entry.namespace), [plugin.GOOGLE_VERTEX_SETTINGS_NAMESPACE])
    const hooks = settings.installs[0]?.hooks
    assert.ok(hooks, 'the plugin installed its settings section')

    const gemini = llm.routes.get(plugin.GEMINI_PROVIDER) as RefreshableAdapter
    const anthropic = llm.routes.get(plugin.PROVIDER) as RefreshableAdapter

    const discovered = await gemini.listModels(plugin.GEMINI_PROVIDER)
    assert.ok(discovered.some(model => model.id === 'gemini-9-live'), 'the first read discovers live models')
    assert.equal(catalog.gemini, 1)
    assert.equal(catalog.anthropic, 0, 'the Anthropic cache was neither asked for nor dropped yet')

    // While the five-minute cache holds, neither route reaches the provider.
    await gemini.listModels(plugin.GEMINI_PROVIDER)
    await anthropic.listModels(plugin.PROVIDER)
    assert.equal(catalog.gemini, 1)
    const probesBefore = catalog.anthropic
    assert.ok(probesBefore > 0, 'the first Anthropic read probes its candidates')
    await anthropic.listModels(plugin.PROVIDER)
    assert.equal(catalog.anthropic, probesBefore)

    // The card's write lands here as a committed change.
    hooks.onChange()

    await gemini.listModels(plugin.GEMINI_PROVIDER)
    await anthropic.listModels(plugin.PROVIDER)
    assert.equal(catalog.gemini, 2, 'Gemini re-discovered after the refresh')
    assert.equal(catalog.anthropic, probesBefore * 2, 'Anthropic re-probed after the refresh')

    await fiber.dispose()
  } finally {
    globalThis.fetch = realFetch
  }
})
