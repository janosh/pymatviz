import { getWithLoc } from "./common.js";
/** Convert for RenderTag */
export function convertRenderTag(node, parent, ctx) {
    const mustache = {
        type: "SvelteRenderTag",
        expression: null,
        parent,
        ...ctx.getConvertLocation(node),
    };
    const callRange = getWithLoc(node.expression);
    ctx.scriptLet.addExpressionFromRange([callRange.start, callRange.end], mustache, null, (expression) => {
        mustache.expression = expression;
        mustache.expression.parent = mustache;
    });
    const atRenderStart = ctx.code.indexOf("@render", mustache.range[0]);
    ctx.addToken("MustacheKeyword", {
        start: atRenderStart,
        end: atRenderStart + 7,
    });
    return mustache;
}
