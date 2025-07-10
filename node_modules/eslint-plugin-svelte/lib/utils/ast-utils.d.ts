import type { ASTNode, RuleContext, SourceCode } from '../types.js';
import type { TSESTree } from '@typescript-eslint/types';
import type { Scope, Variable } from '@typescript-eslint/scope-manager';
import type { AST as SvAST } from 'svelte-eslint-parser';
/**
 * Checks whether or not the tokens of two given nodes are same.
 * @param left A node 1 to compare.
 * @param right A node 2 to compare.
 * @param sourceCode The ESLint source code object.
 * @returns  the source code for the given node.
 */
export declare function equalTokens(left: ASTNode, right: ASTNode, sourceCode: SourceCode): boolean;
/**
 * Get the value of a given node if it's a literal or a template literal.
 */
export declare function getStringIfConstant(node: TSESTree.Expression | TSESTree.PrivateIdentifier): string | null;
/**
 * Check if it need parentheses.
 */
export declare function needParentheses(node: TSESTree.Expression, kind: 'not' | 'logical'): boolean;
/** Checks whether the given node is the html element node or <svelte:element> node. */
export declare function isHTMLElementLike(node: SvAST.SvelteElement | SvAST.SvelteScriptElement | SvAST.SvelteStyleElement): node is SvAST.SvelteHTMLElement | (SvAST.SvelteSpecialElement & {
    name: SvAST.SvelteName & {
        name: 'svelte:element';
    };
});
/**
 * Find the attribute from the given element node
 */
export declare function findAttribute<N extends string>(node: SvAST.SvelteElement | SvAST.SvelteScriptElement | SvAST.SvelteStyleElement | SvAST.SvelteStartTag, name: N): (SvAST.SvelteAttribute & {
    key: SvAST.SvelteAttribute['key'] & {
        name: N;
    };
}) | null;
/**
 * Find the shorthand attribute from the given element node
 */
export declare function findShorthandAttribute<N extends string>(node: SvAST.SvelteElement | SvAST.SvelteScriptElement | SvAST.SvelteStyleElement | SvAST.SvelteStartTag, name: N): (SvAST.SvelteShorthandAttribute & {
    key: SvAST.SvelteShorthandAttribute['key'] & {
        name: N;
    };
}) | null;
/**
 * Find the bind directive from the given element node
 */
export declare function findBindDirective<N extends string>(node: SvAST.SvelteElement | SvAST.SvelteScriptElement | SvAST.SvelteStyleElement | SvAST.SvelteStartTag, name: N): (SvAST.SvelteBindingDirective & {
    key: SvAST.SvelteDirectiveKey & {
        name: SvAST.SvelteDirectiveKey['name'] & {
            name: N;
        };
    };
}) | null;
/**
 * Get the static attribute value from given attribute
 */
export declare function getStaticAttributeValue(node: SvAST.SvelteAttribute): string | null;
/**
 * Get the static attribute value from given attribute
 */
export declare function getLangValue(node: SvAST.SvelteScriptElement | SvAST.SvelteStyleElement): string | null;
/**
 * Find the variable of a given name.
 */
export declare function findVariable(context: RuleContext, node: TSESTree.Identifier): Variable | null;
/**
 * Iterate the identifiers of a given pattern node.
 */
export declare function iterateIdentifiers(node: TSESTree.DestructuringPattern): Iterable<TSESTree.Identifier>;
/**
 * Gets the scope for the current node
 */
export declare function getScope(context: RuleContext, currentNode: TSESTree.Node): Scope;
/** Get the parent node from the given node */
export declare function getParent(node: TSESTree.Node): TSESTree.Node | null;
export type QuoteAndRange = {
    quote: 'unquoted' | 'double' | 'single';
    range: [number, number];
    firstToken: SvAST.Token | SvAST.Comment;
    lastToken: SvAST.Token | SvAST.Comment;
};
/** Get the quote and range from given attribute values */
export declare function getAttributeValueQuoteAndRange(attr: SvAST.SvelteAttribute | SvAST.SvelteDirective | SvAST.SvelteStyleDirective | SvAST.SvelteSpecialDirective, sourceCode: SourceCode): QuoteAndRange | null;
export declare function getMustacheTokens(node: SvAST.SvelteMustacheTag | SvAST.SvelteShorthandAttribute | SvAST.SvelteSpreadAttribute | SvAST.SvelteDebugTag | SvAST.SvelteRenderTag, sourceCode: SourceCode): {
    openToken: SvAST.Token;
    closeToken: SvAST.Token;
};
export declare function getMustacheTokens(node: SvAST.SvelteDirective | SvAST.SvelteSpecialDirective | SvAST.SvelteMustacheTag | SvAST.SvelteShorthandAttribute | SvAST.SvelteSpreadAttribute | SvAST.SvelteDebugTag | SvAST.SvelteRenderTag, sourceCode: SourceCode): {
    openToken: SvAST.Token;
    closeToken: SvAST.Token;
} | null;
/** Get attribute key text */
export declare function getAttributeKeyText(node: SvAST.SvelteAttribute | SvAST.SvelteShorthandAttribute | SvAST.SvelteStyleDirective | SvAST.SvelteDirective | SvAST.SvelteSpecialDirective | SvAST.SvelteGenericsDirective | SvAST.SvelteAttachTag, context: RuleContext): string;
/** Get directive name */
export declare function getDirectiveName(node: SvAST.SvelteDirective): string;
/**
 * Extract all class names used in a HTML element attribute.
 */
export declare function findClassesInAttribute(attribute: SvAST.SvelteAttribute | SvAST.SvelteShorthandAttribute | SvAST.SvelteSpreadAttribute | SvAST.SvelteDirective | SvAST.SvelteStyleDirective | SvAST.SvelteSpecialDirective | SvAST.SvelteGenericsDirective | SvAST.SvelteAttachTag): string[];
/**
 * Returns name of SvelteElement
 */
export declare function getNodeName(node: SvAST.SvelteElement): string;
/**
 * Returns true if element is known void element
 * {@link https://developer.mozilla.org/en-US/docs/Glossary/Empty_element}
 */
export declare function isVoidHtmlElement(node: SvAST.SvelteElement): boolean;
/**
 * Returns true if element is known foreign (SVG or MathML) element
 */
export declare function isForeignElement(node: SvAST.SvelteElement): boolean;
export declare function isSvgElement(node: SvAST.SvelteElement): boolean;
export declare function isMathMLElement(node: SvAST.SvelteElement): boolean;
/** Checks whether the given identifier node is used as an expression. */
export declare function isExpressionIdentifier(node: TSESTree.Identifier): boolean;
/**
 * Finds the variable for a given name in the specified node's scope.
 * Also determines if the replacement name is already in use.
 *
 * If the `name` is set to null, this assumes you're adding a new variable
 * and reports if it is already in use.
 */
export declare function findVariableForReplacement(context: RuleContext, node: TSESTree.Node, name: string | null, replacementName: string): {
    hasConflict: boolean;
    variable: Variable | null;
};
