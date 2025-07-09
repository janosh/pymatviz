import type * as ESTree from "estree";
import type { Comment, Token } from "../ast/index.js";
import type { Context } from "./index.js";
/** Fix locations */
export declare function fixLocations(node: ESTree.Node, tokens: Token[], comments: Comment[], offset: number, visitorKeys: {
    [type: string]: string[];
} | undefined, ctx: Context): void;
