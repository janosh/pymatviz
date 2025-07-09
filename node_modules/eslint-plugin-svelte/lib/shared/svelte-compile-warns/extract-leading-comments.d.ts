import type { AST } from 'svelte-eslint-parser';
import type { RuleContext } from '../../types.js';
import type { ASTNodeWithParent } from '../../types-for-node.js';
/** Extract comments */
export declare function extractLeadingComments(context: RuleContext, node: ASTNodeWithParent): (AST.Token | AST.Comment)[];
