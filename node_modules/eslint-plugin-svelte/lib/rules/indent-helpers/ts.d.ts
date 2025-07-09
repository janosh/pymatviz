import type { IndentContext } from './commons.js';
import type { TSNodeListener } from '../../types-for-node.js';
type NodeListener = TSNodeListener;
/**
 * Creates AST event handlers for svelte nodes.
 *
 * @param context The rule context.
 * @returns AST event handlers.
 */
export declare function defineVisitor(context: IndentContext): NodeListener;
export {};
