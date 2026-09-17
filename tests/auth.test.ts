/**
 * Service-account auth tests: a real RSA key signs a real assertion (verified
 * with the matching public key), and a stubbed token endpoint covers caching,
 * early refresh, and both failure classes.
 *
 * @module dsh-google-vertex/tests/auth
 */

import assert from 'node:assert/strict'
import { generateKeyPairSync, verify } from 'node:crypto'
import test from 'node:test'

import {
  CLOUD_PLATFORM_SCOPE,
  expandHome,
  parseServiceAccount,
  ServiceAccountTokens,
  signedAssertion,
  VertexAuthError,
  type FetchLike,
  type ServiceAccount,
} from '../src/auth.ts'

const { publicKey, privateKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
})

const ACCOUNT: ServiceAccount = {
  client_email: 'test@example.iam.gserviceaccount.com',
  private_key: privateKey,
  private_key_id: 'key-1',
  project_id: 'example-project',
  token_uri: 'https://oauth2.example.test/token',
}

test('expandHome resolves the home shorthand and leaves absolute paths alone', () => {
  assert.match(expandHome('~/.secrets/sa.json'), /\/\.secrets\/sa\.json$/)
  assert.equal(expandHome('/tmp/sa.json'), '/tmp/sa.json')
})

test('parseServiceAccount refuses a document it cannot sign with', () => {
  assert.throws(() => parseServiceAccount('not json', 'f.json'), /not valid JSON/)
  assert.throws(() => parseServiceAccount('[]', 'f.json'), /must contain a JSON object/)
  assert.throws(() => parseServiceAccount('{"client_email":"a@b"}', 'f.json'), /no usable "private_key"/)
  assert.throws(() => parseServiceAccount('{"private_key":"x"}', 'f.json'), /no "client_email"/)
})

test('parseServiceAccount keeps the optional fields it was given', () => {
  const parsed = parseServiceAccount(JSON.stringify(ACCOUNT), 'f.json')
  assert.equal(parsed.client_email, ACCOUNT.client_email)
  assert.equal(parsed.project_id, 'example-project')
  assert.equal(parsed.token_uri, 'https://oauth2.example.test/token')
})

test('signedAssertion is a JWT the matching public key verifies', () => {
  const assertion = signedAssertion(ACCOUNT, 1_700_000_000, CLOUD_PLATFORM_SCOPE)
  const [header, claims, signature] = assertion.split('.')
  assert.ok(header !== undefined && claims !== undefined && signature !== undefined)

  const decodedHeader = JSON.parse(Buffer.from(header, 'base64url').toString()) as Record<string, unknown>
  assert.equal(decodedHeader['alg'], 'RS256')
  assert.equal(decodedHeader['kid'], 'key-1')

  const decodedClaims = JSON.parse(Buffer.from(claims, 'base64url').toString()) as Record<string, unknown>
  assert.equal(decodedClaims['iss'], ACCOUNT.client_email)
  assert.equal(decodedClaims['aud'], ACCOUNT.token_uri)
  assert.equal(decodedClaims['scope'], CLOUD_PLATFORM_SCOPE)
  assert.equal(decodedClaims['iat'], 1_700_000_000)
  assert.equal(decodedClaims['exp'], 1_700_003_600)

  const valid = verify('RSA-SHA256', Buffer.from(`${header}.${claims}`), publicKey, Buffer.from(signature, 'base64url'))
  assert.equal(valid, true)
})

/** A stubbed token endpoint that counts calls and can fail on demand. */
function stubFetch(expiresIn: number): { fetch: FetchLike; calls: () => number; requests: () => RequestInit[] } {
  let calls = 0
  const requests: RequestInit[] = []
  const fetch: FetchLike = (input, init) => {
    calls += 1
    requests.push(init)
    assert.equal(input, 'https://oauth2.example.test/token')
    return Promise.resolve(Response.json({ access_token: `token-${calls}`, expires_in: expiresIn }))
  }
  return { fetch, calls: () => calls, requests: () => requests }
}

test('tokens are minted once and reused until shortly before expiry', async () => {
  const stub = stubFetch(3600)
  let now = 1_000_000
  const tokens = new ServiceAccountTokens(ACCOUNT, { fetch: stub.fetch, now: () => now })

  assert.equal(await tokens.get(), 'token-1')
  assert.equal(await tokens.get(), 'token-1')
  assert.equal(stub.calls(), 1)

  // 54 minutes in: still outside the refresh margin.
  now += 54 * 60 * 1000
  assert.equal(await tokens.get(), 'token-1')
  assert.equal(stub.calls(), 1)

  // 56 minutes in: the margin has been crossed.
  now += 2 * 60 * 1000
  assert.equal(await tokens.get(), 'token-2')
  assert.equal(stub.calls(), 2)

  const body = stub.requests()[0]?.body
  assert.equal(typeof body, 'string')
  assert.match(String(body), /grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer/)
  assert.match(String(body), /assertion=/)
})

test('concurrent callers share one mint', async () => {
  let resolveFetch: (() => void) | undefined
  let calls = 0
  const fetch: FetchLike = async () => {
    calls += 1
    await new Promise<void>((resolve) => { resolveFetch = resolve })
    return Response.json({ access_token: 'shared', expires_in: 3600 })
  }
  const tokens = new ServiceAccountTokens(ACCOUNT, { fetch })
  const waiting = Promise.all([tokens.get(), tokens.get(), tokens.get()])
  resolveFetch?.()
  assert.deepEqual(await waiting, ['shared', 'shared', 'shared'])
  assert.equal(calls, 1)
})

test('a refused credential is an AUTH failure carrying the endpoint detail', async () => {
  const fetch: FetchLike = () => Promise.resolve(
    new Response('{"error":"invalid_grant","error_description":"Invalid JWT Signature."}', { status: 400 }),
  )
  const tokens = new ServiceAccountTokens(ACCOUNT, { fetch })
  await assert.rejects(tokens.get(), (error: unknown) => {
    assert.ok(error instanceof VertexAuthError)
    assert.equal(error.code, 'AUTH')
    assert.match(error.message, /HTTP 400/)
    assert.match(error.message, /Invalid JWT Signature/)
    return true
  })
})

test('an unreachable token endpoint is a TRANSPORT failure', async () => {
  const fetch: FetchLike = () => Promise.reject(new Error('connect ECONNREFUSED'))
  const tokens = new ServiceAccountTokens(ACCOUNT, { fetch })
  await assert.rejects(tokens.get(), (error: unknown) => {
    assert.ok(error instanceof VertexAuthError)
    assert.equal(error.code, 'TRANSPORT')
    assert.match(error.message, /ECONNREFUSED/)
    return true
  })
})

test('a bodiless 200 is refused rather than cached as undefined', async () => {
  const fetch: FetchLike = () => Promise.resolve(Response.json({ token_type: 'Bearer' }))
  const tokens = new ServiceAccountTokens(ACCOUNT, { fetch })
  await assert.rejects(tokens.get(), /without an access_token/)
  // The failure must not poison the cache: a later success still mints.
  const good = new ServiceAccountTokens(ACCOUNT, { fetch: stubFetch(3600).fetch })
  assert.equal(await good.get(), 'token-1')
})
