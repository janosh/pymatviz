import { getDeclaratorFromConstTag } from "../compat.js";
/** Convert for ConstTag */
export function convertConstTag(node, parent, ctx) {
    const mustache = {
        type: "SvelteConstTag",
        declaration: null,
        declarations: [],
        parent,
        ...ctx.getConvertLocation(node),
    };
    // Link declaration and declarations for backward compatibility.
    // TODO Remove in v2 and later.
    Object.defineProperty(mustache, "declaration", {
        get() {
            return mustache.declarations[0];
        },
        set(value) {
            mustache.declarations = [value];
        },
        enumerable: false,
    });
    ctx.scriptLet.addVariableDeclarator(getDeclaratorFromConstTag(node), mustache, (declaration) => {
        mustache.declarations = [declaration];
    });
    const atConstStart = ctx.code.indexOf("@const", mustache.range[0]);
    ctx.addToken("MustacheKeyword", {
        start: atConstStart,
        end: atConstStart + 6,
    });
    return mustache;
}
