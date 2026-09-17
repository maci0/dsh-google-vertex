/**
 * The slice of the DeepSeek Harness host surface this plugin uses, declared
 * structurally.
 *
 * Like the other plugins under `~/dsh-plugins`, this package is installed
 * outside the harness checkout and cannot resolve `@deepseek-ai/*` from its own
 * directory, so it carries no runtime dependency on them. `LlmRuntime` reaches
 * adapters through plain method calls — there is no `instanceof LlmAdapter`
 * check anywhere in it — so a duck-typed adapter is a supported shape.
 *
 * Each declaration is narrowed to what this adapter reads or emits. Fields the
 * adapter never touches are deliberately absent: a mirrored field that nothing
 * reads is a field whose absence goes unnoticed. Widen a declaration when the
 * adapter starts using it, not before.
 *
 * @module dsh-google-vertex/host
 */
export {};
