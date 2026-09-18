/**
 * dsh-google-vertex — browser half: the plugin's configuration page on the
 * Plugins page, under the bundle's `google-vertex` row, and the Refresh control
 * that lives on it.
 *
 * Both adapters on the host half discover their catalogs at runtime behind a
 * five-minute cache, so a model Google published a minute ago is invisible to
 * the model picker until that cache expires. This half has exactly one way to
 * reach the host process — a write to the `google-vertex` settings namespace
 * the host half registers — so the button writes the current time there. The
 * host's change hook drops both cached catalogs, and the settings commit also
 * makes the model picker re-read the catalog from the host. No restart, no
 * harness change.
 *
 * The page contract takes `summary` (the one-liner the page prints under the
 * row's module) and `page` (the controls); the page draws the title and the
 * crumb itself, so this half draws neither. `plugins.row.config` is keyed by
 * `<package name>#<row id>`, which is the key below.
 *
 * This file is plain JavaScript on purpose. The client module system serves a
 * package's `exports["./client"]` artifact as a lazy-CJS factory registered on
 * `window.__ModuleLoader__`, and that is the whole format — an out-of-tree
 * plugin can author it directly instead of reproducing the repository's tsdown
 * client preset. `react` is provided by the module system; nothing else is
 * required here.
 */

window.__ModuleLoader__.load({
  id: 'dsh-google-vertex',

  factory: (require) => {
    var module = { exports: {} }
    var exports = module.exports
    Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' })

    const React = require('react')

    /** Settings namespace shared with the host half. */
    const NAMESPACE = 'google-vertex'

    /** Locale namespace for this plugin's copy. */
    const LOCALE_NS = 'google-vertex'

    /** The Plugins-page slot this card occupies. */
    const SLOT = 'plugins.row.config'

    /**
     * The slot's key: the bundle's package name and the row id its patch
     * declares (`rowConfigKey` in the page's own contract). A deployment that
     * installs this package under another name shows no configure control,
     * which is the whole trace of a mismatch.
     */
    const ENTRY_KEY = 'dsh-google-vertex#google-vertex'

    /** Every class is `gv-`-prefixed: the sheet lands in the page's own document. */
    const CSS = [
      '.gv-page{display:flex;flex-direction:column;gap:10px}',
      '.gv-hint{font-size:12px;line-height:1.5;color:var(--dsw-alias-label-tertiary)}',
      '.gv-row{display:flex;align-items:center;flex-wrap:wrap;gap:10px}',
      '.gv-action{appearance:none;font:inherit;font-size:13px;line-height:1.5;padding:5px 14px;cursor:pointer;color:var(--dsw-alias-label-primary);background:var(--dsw-alias-bg-layer-4);border:1px solid var(--dsw-alias-border-l2);border-radius:999px}',
      '.gv-action:disabled{cursor:default;opacity:.5}',
      '.gv-status{font-size:12px;line-height:1.5;color:var(--dsw-alias-label-tertiary)}',
      '.gv-error{font-size:12px;line-height:1.5;color:var(--dsw-alias-label-error)}',
    ].join('')

    // Appended while the factory materializes: the module system claims the tag
    // for this package and disposes it on unload. Guarded because the node unit
    // tests evaluate this file without a DOM.
    if (typeof document !== 'undefined') {
      const style = document.createElement('style')
      style.textContent = CSS
      document.head.append(style)
    }

    const en = {
      summary: 'Google-hosted Claude and Gemini models, discovered from Vertex at runtime.',
      refresh: 'Refresh models',
      hint: 'Re-reads the catalog from Vertex, then refreshes the model picker. No restart.',
      lastRefresh: 'Last manual refresh: {at}',
      never: 'No manual refresh yet.',
      readOnly: 'Settings are not persisted in this deployment, so this control cannot reach the host. Restart dsh web to re-discover.',
    }

    const zh = {
      summary: '由 Vertex 托管的 Claude 与 Gemini 模型，运行时动态发现。',
      refresh: '刷新模型',
      hint: '从 Vertex 重新读取模型目录，并刷新模型选择器。无需重启。',
      lastRefresh: '上次手动刷新：{at}',
      never: '尚未手动刷新。',
      readOnly: '此部署不持久化设置，该控件无法通知宿主进程。请重启 dsh web 以重新发现。',
    }

    /**
     * Bind one scope to a React subscription.
     * @param scope - a scope bound to the google-vertex settings namespace.
     * @returns a hook reading that scope's current snapshot.
     */
    function useScope(scope) {
      const subscribe = (listener) => scope.subscribe(listener)
      const getSnapshot = () => scope.getSnapshot()
      return () => React.useSyncExternalStore(subscribe, getSnapshot)
    }

    /**
     * Read a snapshot's value. A namespace the deployment does not serve reports
     * no value, which the card renders as nothing at all.
     * @param snapshot - the settings scope snapshot.
     * @returns the stored value object, or `undefined` when unreadable.
     */
    function valueOf(snapshot) {
      if (snapshot.status !== 'ready') return undefined
      return snapshot.value !== null && typeof snapshot.value === 'object' ? snapshot.value : {}
    }

    /**
     * Build the card component over one bound settings scope.
     * @param scope - the scope bound to the google-vertex namespace.
     * @param t - translate function bound to this plugin's locale namespace.
     * @returns the component the page renders in either view.
     */
    function createCard(scope, t) {
      const useSettings = useScope(scope)

      return function GoogleVertexCard({ view }) {
        const snapshot = useSettings()
        const [error, setError] = React.useState(null)

        const value = valueOf(snapshot)
        // A namespace this deployment does not serve renders no trace of itself.
        if (value === undefined) return null
        if (view === 'summary') return t('summary')

        const refresh = () => {
          setError(null)
          Promise.resolve(scope.set('revalidatedAt', new Date().toISOString())).catch((cause) => {
            setError(cause instanceof Error ? cause.message : String(cause))
          })
        }

        const stamp = typeof value.revalidatedAt === 'string' && value.revalidatedAt.length > 0
          ? new Date(value.revalidatedAt)
          : undefined
        const last = stamp !== undefined && !Number.isNaN(stamp.getTime())
          ? t('lastRefresh', { at: stamp.toLocaleString() })
          : t('never')

        return React.createElement(
          'div',
          { className: 'gv-page' },
          React.createElement('p', { className: 'gv-hint' }, t('hint')),
          React.createElement(
            'div',
            { className: 'gv-row' },
            React.createElement(
              'button',
              {
                type: 'button',
                className: 'gv-action',
                disabled: !snapshot.writable,
                onClick: refresh,
              },
              t('refresh'),
            ),
            React.createElement(
              'span',
              { className: 'gv-status' },
              snapshot.writable ? last : t('readOnly'),
            ),
          ),
          error === null ? null : React.createElement('p', { className: 'gv-error' }, error),
        )
      }
    }

    /**
     * Mount the card: this plugin's row configuration on the Plugins page.
     * @param ctx - the browser plugin context.
     */
    function apply(ctx) {
      const t = ctx.locale.bind(LOCALE_NS)
      ctx.effect(
        () => ctx.locale.register(LOCALE_NS, { en, zh }),
        'dsh-google-vertex: locale dictionary',
      )

      const scope = ctx.settingsScope.bind({ namespace: NAMESPACE })
      const Card = createCard(scope, t)

      // The page declares the slot; injecting waits for it to exist, so this
      // registration does not depend on plugin load order.
      ctx.slots.inject(SLOT, () => ctx.slots.register({
        name: SLOT,
        key: ENTRY_KEY,
        locale: LOCALE_NS,
      }, Card))
    }

    exports.apply = apply
    exports.inject = ['slots', 'settingsScope', 'locale']
    return module.exports
  },
})
