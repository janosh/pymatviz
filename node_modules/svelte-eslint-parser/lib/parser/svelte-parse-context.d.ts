import type * as Compiler from "./svelte-ast-types-for-v5.js";
import type * as SvAST from "./svelte-ast-types.js";
import type { NormalizedParserOptions } from "./parser-options.js";
import type { SvelteConfig } from "../svelte-config/index.js";
import type { ESLintProgram } from "./index.js";
/** The context for parsing. */
export type SvelteParseContext = {
    /**
     * Determines if the file is in Runes mode.
     *
     * - Svelte 3/4 does not support Runes mode.
     * - Checks if `runes` configuration exists in:
     *   - `parserOptions`
     *   - `svelte.config.js`
     *   - `<svelte:options>` in the Svelte file.
     * - Returns `true` if the `runes` symbol is present in the Svelte file.
     */
    runes?: boolean;
    /** The version of "svelte/compiler". */
    compilerVersion: string;
    /** The result of static analysis of `svelte.config.js`. */
    svelteConfig: SvelteConfig | null;
};
export declare function resolveSvelteParseContextForSvelte(svelteConfig: SvelteConfig | null, parserOptions: NormalizedParserOptions, svelteAst: Compiler.Root | SvAST.AstLegacy): SvelteParseContext;
export declare function resolveSvelteParseContextForSvelteScript(svelteConfig: SvelteConfig | null): SvelteParseContext;
export declare function hasRunesSymbol(ast: ESLintProgram): boolean;
