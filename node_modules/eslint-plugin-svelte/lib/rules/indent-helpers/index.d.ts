import type { RuleContext, RuleListener } from '../../types.js';
import type { IndentOptions } from './commons.js';
/**
 * Creates AST event handlers for html-indent.
 *
 * @param context The rule context.
 * @param defaultOptions The default value of options.
 * @returns AST event handlers.
 */
export declare function defineVisitor(context: RuleContext, defaultOptions: Partial<IndentOptions>): RuleListener;
