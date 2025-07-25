import type { SvelteDebugTag, SvelteMustacheTag, SvelteMustacheTagRaw, SvelteMustacheTagText } from "../../ast/index.js";
import type { Context } from "../../context/index.js";
import type * as SvAST from "../svelte-ast-types.js";
import type * as Compiler from "../svelte-ast-types-for-v5.js";
/** Convert for MustacheTag */
export declare function convertMustacheTag(node: SvAST.MustacheTag | Compiler.ExpressionTag, parent: SvelteMustacheTag["parent"], typing: string | null, ctx: Context): SvelteMustacheTagText;
/** Convert for MustacheTag */
export declare function convertRawMustacheTag(node: SvAST.RawMustacheTag | Compiler.HtmlTag, parent: SvelteMustacheTag["parent"], ctx: Context): SvelteMustacheTagRaw;
/** Convert for DebugTag */
export declare function convertDebugTag(node: SvAST.DebugTag, parent: SvelteDebugTag["parent"], ctx: Context): SvelteDebugTag;
