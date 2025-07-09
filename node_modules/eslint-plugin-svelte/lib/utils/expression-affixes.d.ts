import type { TSESTree } from '@typescript-eslint/types';
import type { RuleContext } from '../types.js';
import type { AST } from 'svelte-eslint-parser';
export declare function extractExpressionPrefixVariable(context: RuleContext, expression: TSESTree.Expression): TSESTree.Identifier | null;
export declare function extractExpressionPrefixLiteral(context: RuleContext, expression: AST.SvelteLiteral | TSESTree.Node): string | null;
export declare function extractExpressionSuffixLiteral(context: RuleContext, expression: AST.SvelteLiteral | TSESTree.Node): string | null;
