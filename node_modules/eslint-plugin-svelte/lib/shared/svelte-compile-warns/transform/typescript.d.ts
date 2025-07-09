import type { AST } from 'svelte-eslint-parser';
import type { RuleContext } from '../../../types.js';
import type { TransformResult } from './types.js';
/**
 * Transpile with typescript
 */
export declare function transform(node: AST.SvelteScriptElement, text: string, context: RuleContext): TransformResult | null;
/** Check if project has TypeScript. */
export declare function hasTypeScript(context: RuleContext): boolean;
