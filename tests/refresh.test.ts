/**
 * Refresh test: the manual re-discovery the browser half's card asks for.
 *
 * The card can only reach the host process through a settings write, so the
 * whole path is: settings commit → the plugin's change hook → the adapters
 * drop their cached catalogs → the next catalog read reaches Vertex again. Only
 * the Gemini route discovers, so that is the route the stub transport counts;
 * the Claude route serves its configured list and must never reach the network.
 * The test drives that path over a real Cordis `Context`, a stub `settings`
 * service that records the hook, and a stubbed transport.
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
import type { LlmAdapterLike } from '../src/host.ts'

declare module '@deepseek-ai/cordis' {
  interface Events {
    /** Volatile config values were committed into the running fiber; owning fiber only. */
    'loader/volatile-update'(paths: readonly (readonly string[])[]): void
  }
}

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

/** An adapter with the refresh verb the plugin's settings hook calls. */
type RefreshableAdapter = LlmAdapterLike & { invalidateModels(): void }

test('a committed settings write drops the cached catalog, so the next read re-discovers', async () => {
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
  let catalogs = 0
  const realFetch = globalThis.fetch
  globalThis.fetch = ((input: string | URL | Request): Promise<Response> => {
    const url = String(input)
    if (url.includes('oauth2.googleapis.com')) {
      return Promise.resolve(new Response(JSON.stringify({ access_token: 'token', expires_in: 3600 }), { status: 200 }))
    }
    catalogs += 1
    return Promise.resolve(new Response(JSON.stringify({
      models: [{ name: 'publishers/google/models/gemini-9-live', displayName: 'Gemini 9 Live' }],
    }), { status: 200 }))
  }) as typeof globalThis.fetch

  try {
    const ctx = new Context()
    const llm = new StubLlm(ctx)

    const fiber = await ctx.plugin(plugin as unknown as Plugin, { serviceAccountFile: accountPath })

    const gemini = llm.routes.get(plugin.GEMINI_PROVIDER) as RefreshableAdapter
    const anthropic = llm.routes.get(plugin.PROVIDER) as RefreshableAdapter

    const discovered = await gemini.listModels(plugin.GEMINI_PROVIDER)
    assert.ok(discovered.some(model => model.id === 'gemini-9-live'), 'the first read discovers live models')
    assert.equal(catalogs, 1)

    // While the five-minute cache holds, the route does not reach the provider.
    await gemini.listModels(plugin.GEMINI_PROVIDER)
    assert.equal(catalogs, 1)

    // The Claude route serves its configured catalog: no discovery, no request.
    const claude = await anthropic.listModels(plugin.PROVIDER)
    assert.ok(claude.length > 0, 'the Claude catalog is served from configuration')
    assert.equal(catalogs, 1, 'the Claude route made no catalog request')

    // A write of the volatile `revalidatedAt` reaches the plugin as this event.
    ctx.emit('loader/volatile-update', [['revalidatedAt']])

    await gemini.listModels(plugin.GEMINI_PROVIDER)
    assert.equal(catalogs, 2, 'Gemini re-discovered after the refresh')
    assert.deepEqual(await anthropic.listModels(plugin.PROVIDER), claude, 'the Claude catalog is unchanged')

    await fiber.dispose()
  } finally {
    globalThis.fetch = realFetch
  }
})
