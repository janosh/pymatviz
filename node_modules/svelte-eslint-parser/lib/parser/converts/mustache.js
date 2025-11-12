import { hasTypeInfo } from "../../utils/index.js";
/** Convert for MustacheTag */
export function convertMustacheTag(node, parent, typing, ctx) {
    return convertMustacheTag0(node, "text", parent, typing, ctx);
}
/** Convert for MustacheTag */
export function convertRawMustacheTag(node, parent, ctx) {
    const mustache = convertMustacheTag0(node, "raw", parent, null, ctx);
    const atHtmlStart = ctx.code.indexOf("@html", mustache.range[0]);
    ctx.addToken("MustacheKeyword", {
        start: atHtmlStart,
        end: atHtmlStart + 5,
    });
    return mustache;
}
/** Convert for DebugTag */
export function convertDebugTag(node, parent, ctx) {
    const mustache = {
        type: "SvelteDebugTag",
        identifiers: [],
        parent,
        ...ctx.getConvertLocation(node),
    };
    for (const id of node.identifiers) {
        ctx.scriptLet.addExpression(id, mustache, null, (es) => {
            mustache.identifiers.push(es);
        });
    }
    const atDebugStart = ctx.code.indexOf("@debug", mustache.range[0]);
    ctx.addToken("MustacheKeyword", {
        start: atDebugStart,
        end: atDebugStart + 6,
    });
    return mustache;
}
/** Convert to MustacheTag */
function convertMustacheTag0(node, kind, parent, typing, ctx) {
    const mustache = {
        type: "SvelteMustacheTag",
        kind,
        expression: null,
        parent,
        ...ctx.getConvertLocation(node),
    };
    ctx.scriptLet.addExpression(node.expression, mustache, hasTypeInfo(node.expression) ? null : typing, (es) => {
        mustache.expression = es;
    });
    return mustache;
}
