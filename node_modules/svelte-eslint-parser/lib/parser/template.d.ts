import type * as Compiler from "./svelte-ast-types-for-v5.js";
import type * as SvAST from "./svelte-ast-types.js";
import type { Context } from "../context/index.js";
import type { SvelteProgram } from "../ast/index.js";
import type { NormalizedParserOptions } from "./parser-options.js";
/**
 * Parse for template
 */
export declare function parseTemplate(code: string, ctx: Context, parserOptions: NormalizedParserOptions): {
    ast: SvelteProgram;
    svelteAst: Compiler.Root | SvAST.AstLegacy;
};
