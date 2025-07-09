import type { SvelteNodeListener } from '../../types-for-node.js';
import type { IndentContext } from './commons.js';
type NodeListener = SvelteNodeListener;
/**
 * Creates AST event handlers for svelte nodes.
 *
 * @param context The rule context.
 * @returns AST event handlers.
 */
export declare function defineVisitor(context: IndentContext): NodeListener;
export {};
