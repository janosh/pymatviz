import type { SvelteConstTag } from "../../ast/index.js";
import type { Context } from "../../context/index.js";
import type * as SvAST from "../svelte-ast-types.js";
import type * as Compiler from "../svelte-ast-types-for-v5.js";
/** Convert for ConstTag */
export declare function convertConstTag(node: SvAST.ConstTag | Compiler.ConstTag, parent: SvelteConstTag["parent"], ctx: Context): SvelteConstTag;
