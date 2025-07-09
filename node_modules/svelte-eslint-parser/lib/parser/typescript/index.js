import { parseScript, parseScriptInSvelte } from "../script.js";
import { analyzeTypeScript, analyzeTypeScriptInSvelte, } from "./analyze/index.js";
import { setParent } from "./set-parent.js";
/**
 * Parse for TypeScript in <script>
 */
export function parseTypeScriptInSvelte(code, attrs, parserOptions, context) {
    const tsCtx = analyzeTypeScriptInSvelte(code, attrs, parserOptions, context);
    const result = parseScriptInSvelte(tsCtx.script, attrs, parserOptions);
    tsCtx.restoreContext.restore(result);
    return result;
}
/**
 * Parse for TypeScript
 */
export function parseTypeScript(code, attrs, parserOptions, svelteParseContext) {
    const tsCtx = analyzeTypeScript(code, attrs, parserOptions, svelteParseContext);
    const result = parseScript(tsCtx.script, attrs, parserOptions);
    setParent(result);
    tsCtx.restoreContext.restore(result);
    return result;
}
