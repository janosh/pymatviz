import { parse } from "svelte/compiler";
import { convertSvelteRoot } from "./converts/index.js";
import { ParseError } from "../errors.js";
import { svelteVersion } from "./svelte-version.js";
/**
 * Parse for template
 */
export function parseTemplate(code, ctx, parserOptions) {
    try {
        const options = {
            filename: parserOptions.filePath,
            ...(svelteVersion.gte(5) ? { modern: true } : {}),
        };
        const svelteAst = parse(code, options);
        const ast = convertSvelteRoot(svelteAst, ctx);
        return {
            ast,
            svelteAst,
        };
    }
    catch (e) {
        if (typeof e.pos === "number") {
            const err = new ParseError(e.message, e.pos, ctx);
            err.svelteCompilerError = e;
            throw err;
        }
        throw e;
    }
}
