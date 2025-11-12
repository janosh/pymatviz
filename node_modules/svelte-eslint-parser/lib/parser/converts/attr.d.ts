import type { SvelteAttribute, SvelteShorthandAttribute, SvelteAttachTag, SvelteDirective, SvelteSpreadAttribute, SvelteStartTag, SvelteStyleDirective } from "../../ast/index.js";
import type { Context } from "../../context/index.js";
import type * as SvAST from "../svelte-ast-types.js";
import type * as Compiler from "../svelte-ast-types-for-v5.js";
/** Convert for Attributes */
export declare function convertAttributes(attributes: (SvAST.AttributeOrDirective | Compiler.Attribute | Compiler.SpreadAttribute | Compiler.AttachTag | Compiler.Directive)[], parent: SvelteStartTag, ctx: Context): IterableIterator<SvelteAttribute | SvelteShorthandAttribute | SvelteSpreadAttribute | SvelteAttachTag | SvelteDirective | SvelteStyleDirective>;
