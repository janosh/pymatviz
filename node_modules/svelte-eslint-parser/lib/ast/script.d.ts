import type ESTree from "estree";
import type { BaseNode } from "./base.js";
import type { SvelteBindingDirective } from "./html.js";
export type SvelteScriptNode = SvelteReactiveStatement | SvelteFunctionBindingsExpression;
/** Node of `$` statement. */
export interface SvelteReactiveStatement extends BaseNode {
    type: "SvelteReactiveStatement";
    label: ESTree.Identifier & {
        name: "$";
    };
    body: ESTree.Statement;
    parent: ESTree.Node;
}
/** Node of `bind:name={get, set}` expression. */
export interface SvelteFunctionBindingsExpression extends BaseNode {
    type: "SvelteFunctionBindingsExpression";
    expressions: [
        /** Getter */
        ESTree.Expression,
        /** Setter */
        ESTree.Expression
    ];
    parent: SvelteBindingDirective;
}
