import type { SvelteRenderTag } from "../../ast/index.js";
import type { Context } from "../../context/index.js";
import type * as Compiler from "../svelte-ast-types-for-v5.js";
/** Convert for RenderTag */
export declare function convertRenderTag(node: Compiler.RenderTag, parent: SvelteRenderTag["parent"], ctx: Context): SvelteRenderTag;
