import type { Node, Root, Rule } from "postcss";
import { type Node as SelectorNode, type Root as SelectorRoot } from "postcss-selector-parser";
import type { Context } from "../context/index.js";
import type { SourceLocation, SvelteStyleElement } from "../ast/index.js";
export type StyleContext = StyleContextNoStyleElement | StyleContextParseError | StyleContextSuccess | StyleContextUnknownLang;
export interface StyleContextNoStyleElement {
    status: "no-style-element";
}
export interface StyleContextParseError {
    status: "parse-error";
    sourceLang: string;
    error: Error;
}
export interface StyleContextSuccess {
    status: "success";
    sourceLang: string;
    sourceAst: Root;
}
export interface StyleContextUnknownLang {
    status: "unknown-lang";
    sourceLang: string;
}
/**
 * Extracts style source from a SvelteStyleElement and parses it into a PostCSS AST.
 */
export declare function parseStyleContext(styleElement: SvelteStyleElement | undefined, ctx: Context): StyleContext;
/**
 * Parses a PostCSS Rule node's selector and returns its AST.
 */
export declare function parseSelector(rule: Rule): SelectorRoot;
/**
 * Extracts a node location (like that of any ESLint node) from a parsed svelte style node.
 */
export declare function styleNodeLoc(node: Node): Partial<SourceLocation>;
/**
 * Extracts a node range (like that of any ESLint node) from a parsed svelte style node.
 */
export declare function styleNodeRange(node: Node): [number | undefined, number | undefined];
/**
 * Extracts a node location (like that of any ESLint node) from a parsed svelte selector node.
 */
export declare function styleSelectorNodeLoc(node: SelectorNode): Partial<SourceLocation>;
