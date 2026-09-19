# dsh-google-vertex

One Google service-account JSON file unlocks eleven models inside DeepSeek Harness: Claude Opus 4.6, Sonnet 4.6, Opus 4.5, Sonnet 4.5, Haiku 4.5, plus Gemini 3.5 Flash, 3.1 Pro Preview, 3 Flash Preview, 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite.
Install it when your model capacity already lives in a Google Cloud project and you would rather point Harness at a credential file than paste an API key.
Nothing is copied into the harness credential store: the file is read once at mount, and each request trades a signed RS256 assertion for a cached OAuth token.

## What you get

- Two provider routes from one configuration row — `google-vertex-anthropic` for Claude and `google-vertex-gemini` for Gemini — both in the Web model picker.
- Service-account auth with no API key: a path in config, or `GOOGLE_APPLICATION_CREDENTIALS`.
- Streaming with tool calling on both routes. Claude emits raw JSON argument deltas; Gemini emits complete function calls and replays the thought signature Gemini 3 requires.
- Catalogs discovered from Vertex at runtime — Gemini from the publishers catalog API, Claude by probing its candidates — behind a five-minute cache, with the built-in list as the fallback when the provider cannot be reached.
- A **Refresh models** control on this plugin's row page, which drops both caches and re-reads the model picker without restarting `dsh web`.
- Claude prompt-cache breakpoints on tools, system, and the final conversation block, with cache read/write counts reported as usage.
- Failure codes the harness can act on: `429` → `RATE_LIMIT`, `5xx` → `SERVER`, `RESOURCE_EXHAUSTED` → `QUOTA`, oversized → `CONTEXT_WINDOW_EXCEEDED`, a stalled stream → `TIMEOUT`.
- Environment fallbacks for the credential, project, and region.

## Install

```sh
dsh plugin --profile web add github:maci0/dsh-google-vertex
dsh plugin --profile web update dsh-google-vertex   # later, to pull main
```

Then restart `dsh web`. The package declares `dsh.bundle`, so `dsh plugin` appends it to `dsh.profile.bundles` and the package's own `cordis.patch.yml` inserts the `google-vertex` row — there is no row to paste. Adding that row by hand while the package is a bundle mounts the plugin twice, because `insert` does not dedupe ids.

## Configure

Override the row by id in `~/.dsh/profiles/web/cordis.patch.yml`:

```yaml
- id: google-vertex
  config:
    serviceAccountFile: ~/.secrets/global-claude-project.json
    project: global-claude-project
    location: global
```

`~` is expanded, and the file is read once at mount, so a typo fails loudly instead of mid-turn. A patch replaces the targeted row's whole `config`, so restate every key you keep. This file is live-watched: saving it remounts the plugin, no restart.

| Key | Default | Meaning |
|---|---|---|
| `serviceAccountFile` | `$GOOGLE_APPLICATION_CREDENTIALS` | Path to the service-account JSON; `~` is expanded. Nothing is copied into the harness credential store — the file is the credential, and it is read once at mount. |
| `project` | `$GOOGLE_CLOUD_PROJECT`, `$GCLOUD_PROJECT`, then the file's `project_id` | Google Cloud project id, used by both routes. |
| `location` | `$GOOGLE_CLOUD_LOCATION`, then `global` | Region for both endpoints, or `global` for `aiplatform.googleapis.com`. |
| `models` | the Claude catalog below | Claude ids to advertise, replacing the built-in catalog. Any id is accepted at request time regardless. |
| `geminiModels` | the Gemini catalog below | Gemini ids to advertise. A listed id keeps its measured capacities; an unfamiliar one gets the family defaults. |
| `contextWindow` | `200000` | Claude context window reported per model. |
| `maxTokens` | `32000` | Claude output cap applied when a caller omits one. |
| `streamIdleTimeoutMs` | `300000` | Bound on the interval between two stream reads, on both routes. |

Environment variables are read from `process.env` of the harness process.

## Routes and models

Vertex serves the two publishers over different paths, which is why one row registers two routes. The service-account path, project, and region are written once.

| Route | Provider path | Serves |
|---|---|---|
| `google-vertex-anthropic` | `publishers/anthropic/models/{model}:streamRawPredict` | Claude, picker group **Google Vertex AI (Anthropic)** |
| `google-vertex-gemini` | `publishers/google/models/{model}:streamGenerateContent?alt=sse` | Gemini, picker group **Google Vertex AI (Gemini)** |

Claude defaults — every listed family serves 200k context and caps output at 32k unless a caller asks otherwise:

| Picker label | Model id |
|---|---|
| Claude Opus 4.6 (Vertex) | `claude-opus-4-6` |
| Claude Sonnet 4.6 (Vertex) | `claude-sonnet-4-6` |
| Claude Opus 4.5 (Vertex) | `claude-opus-4-5` |
| Claude Sonnet 4.5 (Vertex) | `claude-sonnet-4-5` |
| Claude Haiku 4.5 (Vertex) | `claude-haiku-4-5` |

Gemini defaults — each entry carries its own limits, because Vertex's differ per model:

| Picker label | Model id | Context | Max output |
|---|---|---|---|
| Gemini 3.5 Flash (Vertex) | `gemini-3.5-flash` | 1,048,576 | 65,535 |
| Gemini 3.1 Pro Preview (Vertex) | `gemini-3.1-pro-preview` | 1,048,576 | 65,535 |
| Gemini 3 Flash Preview (Vertex) | `gemini-3-flash-preview` | 1,048,576 | 65,535 |
| Gemini 2.5 Pro (Vertex) | `gemini-2.5-pro` | 1,048,576 | 65,535 |
| Gemini 2.5 Flash (Vertex) | `gemini-2.5-flash` | 1,048,576 | 65,535 |
| Gemini 2.5 Flash-Lite (Vertex) | `gemini-2.5-flash-lite` | 1,048,576 | 65,535 |

The Gemini cap is one below Vertex's own ceiling on purpose: `maxOutputTokens: 65536` is refused with "supported range is from 1 (inclusive) to 65536 (exclusive)".

Ids are Vertex's aliases, so a promoted release needs no edit here. Both catalogs are overridable from configuration for a deployment that pins dated versions.

### Discovery and refresh

The lists above are the fallback, not the whole catalog. Gemini models are read from Vertex's `publishers/google/models` catalog API, and that result is cached for five minutes so opening the model picker is not a network call. The Claude list is served exactly as configured, because Vertex exposes no Anthropic model listing endpoint.

The picker re-reads that catalog when the host says a model input changed, so a model Google published a minute ago stays invisible until then. This plugin's row page carries the control that forces it: open **Plugins** in the sidebar, open the `dsh-google-vertex` bundle, and configure the `google-vertex` row. **Refresh models** drops the cached Gemini catalog and makes the picker re-read it from this process. The card records the last manual refresh in its own `google-vertex` settings namespace, and the write is also the signal — a browser half has no other channel to the host.

## Try it

1. Restart `dsh web`, then open a session.
2. Type `/model` in the composer, or click the model seat beside it.
3. Pick **Google Vertex AI (Anthropic)** → **Claude Sonnet 4.6 (Vertex)**, or the Gemini group for a Gemini model. Selecting a model also makes it the default for new sessions; a session that already sent a request keeps the model recorded in its own log.
4. Type a prompt that needs a tool:

```
List the files in this directory and tell me which one is largest.
```

The agent runs a shell tool and answers from its output. The same works on the Gemini group, including the signed replay of the model's function call.

## How it works

- **Auth.** The service-account JSON is read once at mount. Per request, a JWT is signed with its private key, sent to the file's `token_uri` (or Google's public token endpoint when the file omits one), and traded for an access token (`https://www.googleapis.com/auth/cloud-platform`), which is cached and refreshed five minutes before expiry. The request carries it as a bearer token. A credential problem reports `AUTH`; a token-endpoint transport problem reports `TRANSPORT`.
- **Streaming.** Both routes bound every read by `streamIdleTimeoutMs`. The watchdog owns its own controller, so a stalled read is torn down and the turn ends with a single `TIMEOUT` failure instead of hanging. Caller cancellation ends as an `aborted` finish, not a provider error.
- **Failure classification.** A refused HTTP response and an in-band provider error envelope classify the same way: a named cause first (`RESOURCE_EXHAUSTED` → `QUOTA`, oversized prompt → `CONTEXT_WINDOW_EXCEEDED`), then the numeric code (`429` → `RATE_LIMIT`, `5xx` → `SERVER`, `404` → `NOT_FOUND`). Gemini has no terminal event, so a body that ends without a finish reason is a truncated response unless the watchdog or an in-band error already ended the turn.
- **Replay.** Gemini 3 signs its function calls, and a replay that drops the signature is refused with `400 INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts`. The adapter stores each emitted block's `thoughtSignature` in the harness replay envelope and echoes it on the next request. Signatures are per model, so a cross-model replay sends the call unsigned rather than failing. No `thinkingConfig` is ever sent: Gemini's dynamic-thinking default is what keeps `gemini-2.5-pro` working. Thinking tokens still arrive in `usageMetadata` and count as output.
- **Attribution.** Requests carry `attributionHeaders()` from `@deepseek-ai/dsh-llm` rather than a pinned release line, so `User-Agent` cannot drift from the installed harness.

## Limits

- **Text only.** Both routes declare `inputModalities: ['text']`, so the harness projects images and files to placeholder text before dispatch. Claude and Gemini on Vertex both accept images; wiring the attachment service into these adapters is the work that would add it.
- **No extended thinking on Claude.** That route never sends a `thinking` block, and replaying one needs a thinking signature the harness reasoning block cannot hold. Enabling thinking without it makes every tool round trip fail.
- **Gemini thought summaries are dropped** for the same signature reason, so only answer text is shown.
- **Vertex AI API must be enabled** on the project, and the service account needs `roles/aiplatform.user`.
- **Claude is not servable in every region.** `global`, `us-east5`, and `europe-west1` answered; `us-central1` returned `400 FAILED_PRECONDITION: … is not servable in region us-central1`. A `404 NOT_FOUND` naming the publisher model means the id does not exist on Vertex; `429 RESOURCE_EXHAUSTED` is the project's quota, not a bad request.
- **Claude prompt caching has provider-side minimums.** Blocks under the cacheable minimum are accepted and simply not cached.

### The pi-ai `google-vertex` route

Harness already ships a pi-ai-backed `google-vertex` route that serves Gemini from a credential record carrying the same three environment values. It accepts images and surfaces thinking summaries; this plugin's Gemini route is text-only and drops them. Choose it when those matter. Its record in `~/.dsh/.credentials.yaml` (owner-only, mode 600, live-watched):

```yaml
records:
  llm-pi-ai/google-vertex:
    kind: api-key
    env:
      GOOGLE_APPLICATION_CREDENTIALS: /path/to/service-account.json
      GOOGLE_CLOUD_PROJECT: global-claude-project
      GOOGLE_CLOUD_LOCATION: global
```

Two of its catalog entries disagree with Vertex's own limits, so those models need an override in `~/.dsh/settings.yaml` under `llm-pi-ai.providers.google-vertex.modelOverrides`: `gemini-2.5-pro` needs `reasoningEfforts: false` (Vertex rejects `thinkingBudget: 0` with `400 INVALID_ARGUMENT`), and `gemini-2.5-flash-lite` needs `maxTokens: 65535`. This plugin's own Gemini route needs neither.

## Development

```sh
npm test           # node --test tests/*.test.ts — hermetic, stubbed transport, no network
npm run build      # tsc -p tsconfig.build.json → lib/index.js + lib/types/
npm run typecheck  # tsc -p tsconfig.json
```

The package ships the built `lib/` and declares `dsh.bundle`, so a change to `src/` needs `npm run build` before it takes effect. A profile that installs the package as a local link (`dsh plugin --profile web add link:/path/to/dsh-google-vertex`) picks up a local edit plus that build after a `dsh web` restart; a profile that installs it from a git spec needs the commit pushed and `dsh plugin --profile web update dsh-google-vertex` instead. `lib/client.js` is the exception either way: the browser half is authored directly as plain JavaScript in the client module loader's factory format and is not produced by `tsc`.

Coverage: the signed assertion, token caching and refresh, both auth failure classes, request projection, cache breakpoints, Gemini request projection with signature replay, SSE framing across split chunks, every terminal finish class — including the stream idle bound and an in-band provider error envelope — usage reported only when the provider reported it, and configuration validation. A real Cordis `Context` mount proves both routes are registered and withdrawn with the fiber, and that the plugin's settings change drops both cached catalogs. The browser half is evaluated from `lib/client.js` through the module loader's own registration format, which is how its slot, its Refresh control, and its write are covered without a browser.

Requires Node `^22.19.0 || >=24.0.0`.

## Licence

MIT. See [LICENSE](LICENSE).
