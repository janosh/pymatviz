import type * as SvAST from "../svelte-ast-types.js";
import type * as Compiler from "../svelte-ast-types-for-v5.js";
import type { SvelteProgram } from "../../ast/index.js";
import type { Context } from "../../context/index.js";
/**
 * Convert root
 */
export declare function convertSvelteRoot(svelteAst: Compiler.Root | SvAST.AstLegacy, ctx: Context): SvelteProgram;
