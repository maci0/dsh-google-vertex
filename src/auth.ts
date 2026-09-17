/**
 * Service-account authentication for Google APIs: read one credentials file,
 * sign a JWT with its private key, trade it for an OAuth access token, and
 * cache that token until shortly before it expires.
 *
 * The harness credential plane stores API keys, while a Vertex deployment
 * typically holds a service-account JSON instead — the file is the credential.
 * That is why this module exists rather than a credential reference: nothing in
 * the harness can turn an RSA key into a bearer token on the request path.
 *
 * Everything here is pure enough to test without a network: the token endpoint
 * transport, the clock, and the scope are injectable.
 *
 * @module dsh-google-vertex/auth
 */

import { createSign } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { isAbsolute, join } from 'node:path'

/** Scope every Vertex AI request needs; the token carries nothing narrower. */
export const CLOUD_PLATFORM_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'

/** Token endpoint a service-account file names; this is the public default. */
export const DEFAULT_TOKEN_URI = 'https://oauth2.googleapis.com/token'

/** Assertion lifetime Google accepts; the token endpoint caps it at one hour. */
const ASSERTION_LIFETIME_SECONDS = 3600

/**
 * Refresh this long before the reported expiry so a request cannot start with a
 * token that dies in flight.
 */
const DEFAULT_REFRESH_MARGIN_MS = 5 * 60 * 1000

/** Injectable transport, so tests never touch the network. */
export type FetchLike = (input: string, init: RequestInit) => Promise<Response>

/** The fields of a service-account JSON this module reads. */
export interface ServiceAccount {
  readonly client_email: string
  readonly private_key: string
  readonly private_key_id?: string
  readonly project_id?: string
  readonly token_uri?: string
}

/** An auth failure the adapter reports as `AUTH` or `TRANSPORT`. */
export class VertexAuthError extends Error {
  /** Provider-neutral failure code the adapter forwards. */
  readonly code: 'AUTH' | 'TRANSPORT'

  /**
   * @param code - whether the failure is a credential problem or a transport one.
   * @param message - operator-facing detail.
   */
  constructor(code: 'AUTH' | 'TRANSPORT', message: string) {
    super(message)
    this.name = 'VertexAuthError'
    this.code = code
  }
}

/**
 * Expand a leading `~` to the process home directory.
 *
 * The path arrives from configuration, which is written by a human the same way
 * a shell is: `~/.secrets/vertex.json` means the home directory, not a literal
 * directory named `~`.
 * @param path - configured path, possibly home-relative.
 * @returns an absolute path.
 */
export function expandHome(path: string): string {
  if (path === '~') return homedir()
  if (path.startsWith('~/')) return join(homedir(), path.slice(2))
  return isAbsolute(path) ? path : join(process.cwd(), path)
}

/**
 * Parse a service-account document, refusing one this module cannot sign with.
 *
 * `name` is the path the text came from, so the failure tells an operator which
 * file to fix rather than only which field is wrong.
 * @param raw - the file's text.
 * @param name - source path used in failure messages.
 * @returns the parsed account.
 * @throws {Error} when the document is not JSON or lacks a signing key.
 */
export function parseServiceAccount(raw: string, name: string): ServiceAccount {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error(`google-vertex: service account file ${name} is not valid JSON`)
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error(`google-vertex: service account file ${name} must contain a JSON object`)
  }
  const account = parsed as Record<string, unknown>
  const clientEmail = account['client_email']
  const privateKey = account['private_key']
  if (typeof clientEmail !== 'string' || clientEmail.length === 0) {
    throw new Error(`google-vertex: service account file ${name} has no "client_email"`)
  }
  if (typeof privateKey !== 'string' || !privateKey.includes('BEGIN PRIVATE KEY')) {
    throw new Error(`google-vertex: service account file ${name} has no usable "private_key"`)
  }
  const keyId = account['private_key_id']
  const projectId = account['project_id']
  const tokenUri = account['token_uri']
  return {
    client_email: clientEmail,
    private_key: privateKey,
    ...typeof keyId === 'string' && keyId.length > 0 ? { private_key_id: keyId } : {},
    ...typeof projectId === 'string' && projectId.length > 0 ? { project_id: projectId } : {},
    ...typeof tokenUri === 'string' && tokenUri.length > 0 ? { token_uri: tokenUri } : {},
  }
}

/**
 * Read and parse one service-account file.
 *
 * Read synchronously on purpose: this runs once at plugin mount, where a typo'd
 * path must fail loudly and immediately rather than as an opaque transport error
 * on the first message.
 * @param path - configured path, possibly home-relative.
 * @returns the parsed account.
 * @throws {Error} when the file is unreadable or unusable.
 */
export function loadServiceAccount(path: string): ServiceAccount {
  const resolved = expandHome(path)
  let raw: string
  try {
    raw = readFileSync(resolved, 'utf8')
  } catch (error) {
    throw new Error(
      `google-vertex: cannot read service account file ${resolved}`
      + ` — ${error instanceof Error ? error.message : String(error)}`,
    )
  }
  return parseServiceAccount(raw, resolved)
}

/** base64url of one JSON value, the JWT segment encoding. */
function base64urlJson(value: unknown): string {
  return Buffer.from(JSON.stringify(value), 'utf8').toString('base64url')
}

/**
 * Sign the JWT-bearer assertion for one service account.
 * @param account - the credentials to sign with.
 * @param nowSeconds - current time from the injected clock.
 * @param scope - OAuth scope the token is requested for.
 * @returns the signed assertion.
 */
export function signedAssertion(account: ServiceAccount, nowSeconds: number, scope: string): string {
  const header = {
    alg: 'RS256',
    typ: 'JWT',
    ...account.private_key_id === undefined ? {} : { kid: account.private_key_id },
  }
  const claims = {
    iss: account.client_email,
    scope,
    aud: account.token_uri ?? DEFAULT_TOKEN_URI,
    iat: nowSeconds,
    exp: nowSeconds + ASSERTION_LIFETIME_SECONDS,
  }
  const signingInput = `${base64urlJson(header)}.${base64urlJson(claims)}`
  const signer = createSign('RSA-SHA256')
  signer.update(signingInput)
  const signature = signer.sign(account.private_key, 'base64url')
  return `${signingInput}.${signature}`
}

/** One cached token and the moment it stops being usable. */
interface CachedToken {
  value: string
  expiresAt: number
}

/** Options for {@link ServiceAccountTokens}. */
export interface TokenSourceOptions {
  /** Transport; defaults to the process `fetch`. */
  fetch?: FetchLike
  /** Clock in milliseconds; defaults to `Date.now`. */
  now?: () => number
  /** OAuth scope; defaults to {@link CLOUD_PLATFORM_SCOPE}. */
  scope?: string
  /** Early-refresh margin in milliseconds. */
  refreshMarginMs?: number
}

/**
 * Access tokens for one service account, refreshed on demand.
 *
 * Concurrent callers share one mint in flight, so a first turn with parallel
 * requests does not sign one assertion per request.
 */
export class ServiceAccountTokens {
  readonly #account: ServiceAccount
  readonly #fetch: FetchLike
  readonly #now: () => number
  readonly #scope: string
  readonly #refreshMarginMs: number
  #cached: CachedToken | undefined
  #pending: Promise<string> | undefined

  /**
   * @param account - the parsed service-account credentials.
   * @param options - injectable transport, clock, scope, and refresh margin.
   */
  constructor(account: ServiceAccount, options: TokenSourceOptions = {}) {
    this.#account = account
    this.#fetch = options.fetch ?? ((input, init) => globalThis.fetch(input, init))
    this.#now = options.now ?? (() => Date.now())
    this.#scope = options.scope ?? CLOUD_PLATFORM_SCOPE
    this.#refreshMarginMs = options.refreshMarginMs ?? DEFAULT_REFRESH_MARGIN_MS
  }

  /**
   * A currently valid access token, minting one when the cache is cold or stale.
   * @param signal - cancellation for the first caller's mint; waiters attached
   *   to an in-flight mint are cancelled only by that mint, which is the
   *   accepted ceiling of sharing one token (a per-caller mint would remove it).
   * @returns the bearer token.
   * @throws {VertexAuthError} `AUTH` for a refused credential, `TRANSPORT` when
   *   the token endpoint could not be reached or answered unusably.
   */
  async get(signal?: AbortSignal): Promise<string> {
    const cached = this.#cached
    if (cached !== undefined && this.#now() < cached.expiresAt - this.#refreshMarginMs) return cached.value

    const pending = this.#pending ?? this.#mint(signal)
    this.#pending = pending
    try {
      return await pending
    } finally {
      if (this.#pending === pending) this.#pending = undefined
    }
  }

  /** Mint one token and cache it. */
  async #mint(signal?: AbortSignal): Promise<string> {
    const nowSeconds = Math.floor(this.#now() / 1000)
    const assertion = signedAssertion(this.#account, nowSeconds, this.#scope)
    const tokenUri = this.#account.token_uri ?? DEFAULT_TOKEN_URI

    let response: Response
    try {
      response = await this.#fetch(tokenUri, {
        method: 'POST',
        headers: { 'content-type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
          assertion,
        }).toString(),
        ...signal === undefined ? {} : { signal },
      })
    } catch (error) {
      if (signal?.aborted === true) {
        throw new VertexAuthError('TRANSPORT', 'google-vertex: token request aborted')
      }
      throw new VertexAuthError(
        'TRANSPORT',
        `google-vertex: token endpoint unreachable — ${error instanceof Error ? error.message : String(error)}`,
      )
    }

    const text = await response.text().catch(() => '')
    if (!response.ok) {
      throw new VertexAuthError(
        'AUTH',
        `google-vertex: token endpoint refused the service account (HTTP ${response.status})`
        + `${text.length > 0 ? ` — ${text.slice(0, 300)}` : ''}`,
      )
    }
    let payload: unknown
    try {
      payload = JSON.parse(text)
    } catch {
      throw new VertexAuthError('TRANSPORT', 'google-vertex: token endpoint answered with a non-JSON body')
    }
    const fields = (typeof payload === 'object' && payload !== null ? payload : {}) as Record<string, unknown>
    const token = fields['access_token']
    if (typeof token !== 'string' || token.length === 0) {
      throw new VertexAuthError('TRANSPORT', `google-vertex: token endpoint answered without an access_token`)
    }
    const expiresIn = typeof fields['expires_in'] === 'number' && fields['expires_in'] > 0
      ? fields['expires_in']
      : ASSERTION_LIFETIME_SECONDS
    this.#cached = { value: token, expiresAt: this.#now() + expiresIn * 1000 }
    return token
  }
}
