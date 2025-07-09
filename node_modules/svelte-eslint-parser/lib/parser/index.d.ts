import type { Comment, SourceLocation, SvelteProgram, Token } from "../ast/index.js";
import type { Program } from "estree";
import type { ScopeManager } from "eslint-scope";
import type { Rule, Node } from "postcss";
import type { Node as SelectorNode, Root as SelectorRoot } from "postcss-selector-parser";
import type * as SvAST from "./svelte-ast-types.js";
import type * as Compiler from "./svelte-ast-types-for-v5.js";
import { type StyleContext, type StyleContextNoStyleElement, type StyleContextParseError, type StyleContextSuccess, type StyleContextUnknownLang } from "./style-context.js";
import { type SvelteParseContext } from "./svelte-parse-context.js";
export { StyleContext, StyleContextNoStyleElement, StyleContextParseError, StyleContextSuccess, StyleContextUnknownLang, };
export interface ESLintProgram extends Program {
    comments: Comment[];
    tokens: Token[];
}
/**
 * The parsing result of ESLint custom parsers.
 */
export interface ESLintExtendedProgram {
    ast: ESLintProgram;
    services?: Record<string, any>;
    visitorKeys?: {
        [type: string]: string[];
    };
    scopeManager?: ScopeManager;
    _virtualScriptCode?: string;
}
type ParseResult = {
    ast: SvelteProgram;
    services: Record<string, any> & ({
        isSvelte: true;
        isSvelteScript: false;
        getSvelteHtmlAst: () => SvAST.Fragment | Compiler.Fragment;
        getStyleContext: () => StyleContext;
        getStyleSelectorAST: (rule: Rule) => SelectorRoot;
        styleNodeLoc: (node: Node) => Partial<SourceLocation>;
        styleNodeRange: (node: Node) => [number | undefined, number | undefined];
        styleSelectorNodeLoc: (node: SelectorNode) => Partial<SourceLocation>;
        svelteParseContext: SvelteParseContext;
    } | {
        isSvelte: false;
        isSvelteScript: true;
        svelteParseContext: SvelteParseContext;
    });
    visitorKeys: {
        [type: string]: string[];
    };
    scopeManager: ScopeManager;
};
/**
 * Parse source code
 */
export declare function parseForESLint(code: string, options?: any): ParseResult;
