import type { RuleModule, PartialRuleModule, PartialRuleMetaData } from '../types.js';
import { type SvelteContext } from '../utils/svelte-context.js';
export declare function shouldRun(svelteContext: SvelteContext | null, conditions: PartialRuleMetaData['conditions']): boolean;
/**
 * Define the rule.
 * @param ruleName ruleName
 * @param rule rule module
 */
export declare function createRule(ruleName: string, rule: PartialRuleModule): RuleModule;
