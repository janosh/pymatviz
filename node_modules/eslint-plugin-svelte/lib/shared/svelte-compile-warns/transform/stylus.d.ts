import type { AST } from 'svelte-eslint-parser';
import type { RuleContext } from '../../../types.js';
import type { TransformResult } from './types.js';
/**
 * Transpile with stylus
 */
export declare function transform(node: AST.SvelteStyleElement, text: string, context: RuleContext): TransformResult | null;
