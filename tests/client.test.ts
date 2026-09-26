/**
 * Browser-half tests: the bundle is plain JavaScript in the client module
 * loader's factory format, so it is evaluated here the same way the browser
 * module system evaluates it — through a minimal `window.__ModuleLoader__` and
 * a React stub whose hooks can be re-rendered after a click.
 *
 * @module dsh-google-vertex/tests/client
 */

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const bundlePath = join(packageRoot, 'lib', 'client.js')

interface Element {
  type: unknown
  props: Record<string, unknown>
  children: unknown[]
}

/** Minimal React stub with stateful hooks, so a click can be re-rendered. */
function createReactStub() {
  const hooks: unknown[] = []
  let cursor = 0
  return {
    reset: (): void => { cursor = 0 },
    createElement: (type: unknown, props: Record<string, unknown> | null, ...children: unknown[]): Element => ({
      type,
      props: props ?? {},
      children: children.flat(),
    }),
    useSyncExternalStore: (_subscribe: () => void, getSnapshot: () => unknown): unknown => getSnapshot(),
    useState: (initial: unknown): [unknown, (next: unknown) => void] => {
      const slot = cursor
      cursor += 1
      if (hooks.length <= slot) hooks[slot] = initial
      return [hooks[slot], (next: unknown) => {
        hooks[slot] = typeof next === 'function' ? (next as (prev: unknown) => unknown)(hooks[slot]) : next
      }]
    },
  }
}

type ReactStub = ReturnType<typeof createReactStub>

interface Snapshot {
  status: string
  value: unknown
  user: unknown
  writable: boolean
}

type Component = (props: { view: 'summary' | 'page' }) => Element | string | null

/** Load the bundle the way the client module system does and return its exports. */
function loadBundle(snapshot: Snapshot, writes: unknown[][]) {
  const react = createReactStub()
  const scope = {
    subscribe: (): (() => void) => () => {},
    getSnapshot: (): Snapshot => snapshot,
    set: async (field: string, value: unknown): Promise<void> => { writes.push([field, value]) },
  }

  // One dictionary per namespace, registered `en` first, with the service's own
  // `{name}` interpolation.
  const dictionaries: Record<string, Record<string, string>> = {}
  const translate = (ns: string, key: string, params?: Record<string, unknown>): string => {
    const template = dictionaries[ns]?.[key] ?? key
    if (params === undefined) return template
    return template.replace(/\{(\w+)\}/g, (match, name: string) =>
      name in params ? String(params[name]) : match)
  }
  const registeredLocales: string[] = []

  const bound: string[] = []
  const injected: string[] = []
  const registered: { entry: Record<string, unknown>; component: Component }[] = []
  const ctx = {
    configForms: { get: (namespace: string) => { bound.push(namespace); return scope } },
    locale: {
      register: (ns: string, dicts: Record<string, Record<string, string>>): (() => void) => {
        registeredLocales.push(ns)
        dictionaries[ns] = dicts['en'] ?? {}
        return () => {}
      },
      bind: (ns: string) => (key: string, params?: Record<string, unknown>) => translate(ns, key, params),
    },
    effect: (callback: () => unknown): (() => void) => { callback(); return () => {} },
    slots: {
      inject: (name: string, callback: () => unknown): void => { injected.push(name); callback() },
      register: (entry: Record<string, unknown>, component: Component) => {
        registered.push({ entry, component })
        return () => {}
      },
    },
  }

  let loaded: { id: string; factory: (require: (id: string) => unknown) => Record<string, unknown> } | undefined
  const windowStub = { __ModuleLoader__: { load: (registration: typeof loaded): void => { loaded = registration } } }
  const requireFn = (id: string): unknown => {
    assert.equal(id, 'react', `the bundle may only require react, got ${id}`)
    return react
  }

  new Function('window', 'require', readFileSync(bundlePath, 'utf8'))(windowStub, requireFn)
  assert.ok(loaded, 'the bundle registered itself on window.__ModuleLoader__')
  assert.equal(loaded.id, 'dsh-google-vertex')

  const exported = loaded.factory(requireFn)
  ;(exported['apply'] as (ctx: unknown) => void)(ctx)
  return { exported, bound, injected, registered, registeredLocales, react }
}

/** Collect every element in a rendered tree. */
function walk(node: unknown, found: Element[] = []): Element[] {
  if (node === null || typeof node !== 'object') return found
  if (Array.isArray(node)) {
    for (const child of node) walk(child, found)
    return found
  }
  const element = node as Element
  if ('props' in element && 'type' in element) {
    found.push(element)
    for (const child of element.children) walk(child, found)
  }
  return found
}

/** Render the card in one view through the stub, resetting its hook cursor. */
function render(react: ReactStub, component: Component, view: 'summary' | 'page'): Element[] {
  react.reset()
  return walk(component({ view }))
}

/** The raw return of one view: an element tree for `page`, a string for `summary`. */
function raw(react: ReactStub, component: Component, view: 'summary' | 'page'): unknown {
  react.reset()
  return component({ view })
}

/** The text of every single-string element, in render order. */
function text(tree: Element[]): unknown[] {
  return tree
    .filter((element) => typeof element.children[0] === 'string' && element.children.length === 1)
    .map((element) => element.children[0])
}

/** A ready snapshot for one stored value. */
function ready(value: unknown, writable = true): Snapshot {
  return { status: 'ready', value, user: {}, writable }
}

/** The one card the bundle registers. */
function cardOf(registered: { entry: Record<string, unknown>; component: Component }[]): Component {
  assert.equal(registered.length, 1)
  const component = registered[0]?.component
  assert.ok(component)
  return component
}

test('the card claims the row slot for the google-vertex row and binds the shared namespace', () => {
  const { exported, bound, injected, registered, registeredLocales } = loadBundle(ready({ revalidatedAt: '' }), [])

  assert.deepEqual(exported['inject'], ['slots', 'configForms', 'locale'])
  assert.deepEqual(bound, ['google-vertex'])
  assert.deepEqual(injected, ['plugins.row.config'])
  assert.deepEqual(registeredLocales, ['google-vertex'])
  assert.equal(registered[0]?.entry['name'], 'plugins.row.config')
  // `rowConfigKey(<package name>, <row id>)`: the configure control appears on
  // the row this key names, and nowhere else.
  assert.equal(registered[0]?.entry['key'], 'dsh-google-vertex#google-vertex')
  assert.equal(registered[0]?.entry['locale'], 'google-vertex')
})

test('the summary view is the one-liner the page prints under the row', () => {
  const { registered, react } = loadBundle(ready({ revalidatedAt: '' }), [])

  assert.equal(
    raw(react, cardOf(registered), 'summary'),
    'Google-hosted Claude and Gemini models, discovered from Vertex at runtime.',
  )
})

test('the page view offers one manual refresh, which records the write in the namespace', () => {
  const writes: unknown[][] = []
  const { registered, react } = loadBundle(ready({ revalidatedAt: '' }), writes)

  const tree = render(react, cardOf(registered), 'page')
  const buttons = tree.filter((element) => element.type === 'button')
  assert.equal(buttons.length, 1)
  const refresh = buttons[0]
  assert.ok(refresh)
  assert.equal(refresh.children[0], 'Refresh models')
  assert.equal(refresh.props['disabled'], false)
  assert.ok(text(tree).includes('Re-reads the catalog from Vertex, then refreshes the model picker. No restart.'))

  const before = Date.now()
  ;(refresh.props['onClick'] as () => void)()
  assert.equal(writes.length, 1)
  assert.equal(writes[0]?.[0], 'revalidatedAt')
  const stamp = Date.parse(String(writes[0]?.[1]))
  assert.ok(Number.isFinite(stamp) && stamp >= before, `expected a fresh timestamp, got ${String(writes[0]?.[1])}`)

  // No inline styles: the sheet carries every rule, so React's style diffing
  // never sees a removed longhand decompose a shorthand.
  for (const element of tree) {
    assert.equal(element.props['style'], undefined, `${String(element.type)} carries an inline style`)
    assert.equal(typeof element.props['className'], 'string', `${String(element.type)} carries no class`)
  }
})

test('the card reports the last manual refresh, or that there has never been one', () => {
  const never = loadBundle(ready({ revalidatedAt: '' }), [])
  const neverCard = cardOf(never.registered)
  assert.ok(text(render(never.react, neverCard, 'page')).includes('No manual refresh yet.'))

  const at = new Date('2026-09-18T10:00:00.000Z')
  const done = loadBundle(ready({ revalidatedAt: at.toISOString() }), [])
  const doneCard = cardOf(done.registered)
  assert.ok(
    text(render(done.react, doneCard, 'page')).includes(`Last manual refresh: ${at.toLocaleString()}`),
    'the stored timestamp is rendered in the reader`s own locale',
  )
})

test('a read-only document disables the control and says why', () => {
  const { registered, react } = loadBundle(ready({ revalidatedAt: '' }, false), [])

  const tree = render(react, cardOf(registered), 'page')
  assert.equal(tree.filter((element) => element.type === 'button')[0]?.props['disabled'], true)
  assert.ok(text(tree).some((line) => typeof line === 'string' && line.includes('Restart dsh web')))
})

test('an unavailable namespace renders no trace of the card in either view', () => {
  const { registered, react } = loadBundle(
    { status: 'loading', value: undefined, user: undefined, writable: false },
    [],
  )

  const component = cardOf(registered)
  assert.equal(raw(react, component, 'summary'), null)
  assert.equal(raw(react, component, 'page'), null)
})
