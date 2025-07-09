import { analyzeScope } from "./analyze-scope.js";
import { traverseNodes } from "../traverse.js";
import { getParser } from "./resolve-parser.js";
import { isEnhancedParserObject } from "./parser-object.js";
/**
 * Parse for <script>
 */
export function parseScriptInSvelte(code, attrs, parserOptions) {
    const result = parseScript(code, attrs, parserOptions);
    traverseNodes(result.ast, {
        visitorKeys: result.visitorKeys,
        enterNode(node, parent) {
            node.parent = parent;
            if (node.type === "LabeledStatement" && node.label.name === "$") {
                if (parent?.type === "Program") {
                    // Transform node type
                    node.type = "SvelteReactiveStatement";
                }
            }
        },
        leaveNode() {
            //
        },
    });
    return result;
}
/**
 * Parse for script
 */
export function parseScript(code, attrs, parserOptions) {
    const result = parseScriptWithoutAnalyzeScopeFromVCode(code, attrs, parserOptions);
    if (!result.scopeManager) {
        const scopeManager = analyzeScope(result.ast, parserOptions);
        result.scopeManager = scopeManager;
    }
    return result;
}
/**
 * Parse for script without analyze scope
 */
export function parseScriptWithoutAnalyzeScope(code, attrs, options) {
    const parser = getParser(attrs, options.parser);
    const result = isEnhancedParserObject(parser)
        ? parser.parseForESLint(code, options)
        : parser.parse(code, options);
    if ("ast" in result && result.ast != null) {
        return result;
    }
    return { ast: result };
}
/**
 * Parse for script without analyze scope
 */
function parseScriptWithoutAnalyzeScopeFromVCode(code, attrs, options) {
    const result = parseScriptWithoutAnalyzeScope(code, attrs, options);
    result._virtualScriptCode = code;
    return result;
}
