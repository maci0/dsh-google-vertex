/**
 * Real-composition test: the plugin mounted into a real `@deepseek-ai/cordis`
 * `Context` over a minimal `llm` service and a launcher environment snapshot.
 *
 * Nothing in the plugin records its own registration — the route list and its
 * withdrawal are the service's business — so the stub owns routes through the
 * calling fiber's effect, exactly as `LlmRuntime.registerAdapter` does. The
 * environment assertion covers the other seam: the plugin must read the
 * launcher's snapshot rather than the flattened `process.env`.
 *
 * @module dsh-google-vertex/tests/composition
 */

import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { Context, Service } from '@deepseek-ai/cordis'
import type { Plugin } from '@deepseek-ai/cordis'
import { createLaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'

import * as plugin from '../src/index.ts'
import type { LlmAdapterLike } from '../src/host.ts'

/**
 * The narrowest stand-in for `ctx.llm`. `this.ctx` traces to the calling fiber,
 * so a route registered from the plugin's `apply` is released when that fiber
 * disposes.
 */
class StubLlm extends Service {
  /** Provider route → adapter, as the service holds them. */
  readonly routes = new Map<string, unknown>()

  constructor(ctx: Context) {
    super(ctx, 'llm')
  }

  /**
   * @param providers - routes this adapter serves.
   * @param adapter - the adapter object.
   * @returns the registration's disposer.
   */
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

test('mounting the plugin on a real Context registers both routes, reads the launch environment, and withdraws on disposal', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-google-vertex-composition-'))
  const accountPath = join(dir, 'service-account.json')
  writeFileSync(accountPath, JSON.stringify({
    type: 'service_account',
    // Deliberately not the project the snapshot names, so the assertion below
    // can only pass when the launch environment supplied it.
    project_id: 'credentials-file-project',
    client_email: 'sa@example.iam.gserviceaccount.com',
    private_key: '-----BEGIN PRIVATE KEY-----\nunused-in-this-test\n-----END PRIVATE KEY-----\n',
    token_uri: 'https://oauth2.googleapis.com/token',
  }))

  const ctx = new Context()
  const llm = new StubLlm(ctx)
  ctx.provide('launchEnvironment', createLaunchEnvironmentSnapshot([
    { source: 'project-env', path: join(dir, '.env'), values: { GOOGLE_CLOUD_PROJECT: 'snapshot-project-77' } },
  ]))

  const fiber = await ctx.plugin(plugin as unknown as Plugin, { serviceAccountFile: accountPath })
  assert.deepEqual([...llm.routes.keys()], [plugin.PROVIDER, plugin.GEMINI_PROVIDER])
  assert.equal(llm.routes.get(plugin.PROVIDER)?.constructor.name, 'GoogleVertexAnthropicAdapter')
  assert.equal(llm.routes.get(plugin.GEMINI_PROVIDER)?.constructor.name, 'GoogleVertexGeminiAdapter')

  // The adapter's own metadata names the project it resolved, which is how the
  // snapshot's value is observable without reading the mount log.
  const anthropic = llm.routes.get(plugin.PROVIDER) as LlmAdapterLike
  const [listed] = await anthropic.listModels(plugin.PROVIDER)
  assert.match(String(listed?.description), /snapshot-project-77/)

  await fiber.dispose()
  assert.deepEqual([...llm.routes.keys()], [])
})
