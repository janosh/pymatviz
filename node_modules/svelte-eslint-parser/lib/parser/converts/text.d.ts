import type { SvelteLiteral, SvelteText } from "../../ast/index.js";
import type { Context } from "../../context/index.js";
import type * as SvAST from "../svelte-ast-types.js";
/** Convert for Text */
export declare function convertText(node: SvAST.Text, parent: SvelteText["parent"], ctx: Context): SvelteText;
/** Convert for Text to Literal */
export declare function convertTextToLiteral(node: SvAST.Text, parent: SvelteLiteral["parent"], ctx: Context): SvelteLiteral;
