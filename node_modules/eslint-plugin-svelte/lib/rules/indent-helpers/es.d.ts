import type { IndentContext } from './commons.js';
import type { ESNodeListener } from '../../types-for-node.js';
type NodeListener = ESNodeListener;
/**
 * Creates AST event handlers for ES nodes.
 *
 * @param context The rule context.
 * @returns AST event handlers.
 */
export declare function defineVisitor(context: IndentContext): NodeListener;
export {};
