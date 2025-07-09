import type ESTree from "estree";
import type { ScopeManager } from "eslint-scope";
import type { SvelteScriptElement, SvelteSnippetBlock } from "../ast/index.js";
import type { NormalizedParserOptions } from "./parser-options.js";
import type { SvelteParseContext } from "./svelte-parse-context.js";
/**
 * Analyze scope
 */
export declare function analyzeScope(node: ESTree.Node, parserOptions: NormalizedParserOptions): ScopeManager;
/** Analyze reactive scope */
export declare function analyzeReactiveScope(scopeManager: ScopeManager): void;
/**
 * Analyze store scope. e.g. $count
 */
export declare function analyzeStoreScope(scopeManager: ScopeManager, svelteParseContext: SvelteParseContext): void;
/** Transform props exports */
export declare function analyzePropsScope(body: SvelteScriptElement, scopeManager: ScopeManager, svelteParseContext: SvelteParseContext): void;
/** Analyze snippets in component scope */
export declare function analyzeSnippetsScope(snippets: SvelteSnippetBlock[], scopeManager: ScopeManager): void;
