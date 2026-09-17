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
/** Scope every Vertex AI request needs; the token carries nothing narrower. */
export declare const CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
/** Token endpoint a service-account file names; this is the public default. */
export declare const DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token";
/** Injectable transport, so tests never touch the network. */
export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;
/** The fields of a service-account JSON this module reads. */
export interface ServiceAccount {
    readonly client_email: string;
    readonly private_key: string;
    readonly private_key_id?: string;
    readonly project_id?: string;
    readonly token_uri?: string;
}
/** An auth failure the adapter reports as `AUTH` or `TRANSPORT`. */
export declare class VertexAuthError extends Error {
    /** Provider-neutral failure code the adapter forwards. */
    readonly code: 'AUTH' | 'TRANSPORT';
    /**
     * @param code - whether the failure is a credential problem or a transport one.
     * @param message - operator-facing detail.
     */
    constructor(code: 'AUTH' | 'TRANSPORT', message: string);
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
export declare function expandHome(path: string): string;
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
export declare function parseServiceAccount(raw: string, name: string): ServiceAccount;
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
export declare function loadServiceAccount(path: string): ServiceAccount;
/**
 * Sign the JWT-bearer assertion for one service account.
 * @param account - the credentials to sign with.
 * @param nowSeconds - current time from the injected clock.
 * @param scope - OAuth scope the token is requested for.
 * @returns the signed assertion.
 */
export declare function signedAssertion(account: ServiceAccount, nowSeconds: number, scope: string): string;
/** Options for {@link ServiceAccountTokens}. */
export interface TokenSourceOptions {
    /** Transport; defaults to the process `fetch`. */
    fetch?: FetchLike;
    /** Clock in milliseconds; defaults to `Date.now`. */
    now?: () => number;
    /** OAuth scope; defaults to {@link CLOUD_PLATFORM_SCOPE}. */
    scope?: string;
    /** Early-refresh margin in milliseconds. */
    refreshMarginMs?: number;
}
/**
 * Access tokens for one service account, refreshed on demand.
 *
 * Concurrent callers share one mint in flight, so a first turn with parallel
 * requests does not sign one assertion per request.
 */
export declare class ServiceAccountTokens {
    #private;
    /**
     * @param account - the parsed service-account credentials.
     * @param options - injectable transport, clock, scope, and refresh margin.
     */
    constructor(account: ServiceAccount, options?: TokenSourceOptions);
    /**
     * A currently valid access token, minting one when the cache is cold or stale.
     * @param signal - cancellation for the first caller's mint; waiters attached
     *   to an in-flight mint are cancelled only by that mint, which is the
     *   accepted ceiling of sharing one token (a per-caller mint would remove it).
     * @returns the bearer token.
     * @throws {VertexAuthError} `AUTH` for a refused credential, `TRANSPORT` when
     *   the token endpoint could not be reached or answered unusably.
     */
    get(signal?: AbortSignal): Promise<string>;
}
