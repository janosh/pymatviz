import { KEYS } from "../visitor-keys.js";
import { Context } from "../context/index.js";
import { Variable } from "eslint-scope";
import { parseScript, parseScriptInSvelte } from "./script.js";
import { sortNodes } from "./sort.js";
import { parseTemplate } from "./template.js";
import { analyzePropsScope, analyzeReactiveScope, analyzeSnippetsScope, analyzeStoreScope, } from "./analyze-scope.js";
import { ParseError } from "../errors.js";
import { parseTypeScript, parseTypeScriptInSvelte, } from "./typescript/index.js";
import { addReference } from "../scope/index.js";
import { parseStyleContext, parseSelector, styleNodeLoc, styleNodeRange, styleSelectorNodeLoc, } from "./style-context.js";
import { getGlobalsForSvelte, getGlobalsForSvelteScript } from "./globals.js";
import { isTypeScript, normalizeParserOptions } from "./parser-options.js";
import { getFragmentFromRoot } from "./compat.js";
import { hasRunesSymbol, resolveSvelteParseContextForSvelte, resolveSvelteParseContextForSvelteScript, } from "./svelte-parse-context.js";
import { resolveSvelteConfigFromOption } from "../svelte-config/index.js";
/**
 * Parse source code
 */
export function parseForESLint(code, options) {
    const svelteConfig = resolveSvelteConfigFromOption(options);
    const parserOptions = normalizeParserOptions(options);
    if (parserOptions.filePath &&
        (parserOptions.filePath.endsWith(".svelte.js") ||
            parserOptions.filePath.endsWith(".svelte.ts"))) {
        const svelteParseContext = resolveSvelteParseContextForSvelteScript(svelteConfig);
        return parseAsScript(code, parserOptions, svelteParseContext);
    }
    return parseAsSvelte(code, svelteConfig, parserOptions);
}
/**
 * Parse source code as svelte component
 */
function parseAsSvelte(code, svelteConfig, parserOptions) {
    const ctx = new Context(code, parserOptions);
    const resultTemplate = parseTemplate(ctx.sourceCode.template, ctx, parserOptions);
    const svelteParseContext = resolveSvelteParseContextForSvelte(svelteConfig, parserOptions, resultTemplate.svelteAst);
    const scripts = ctx.sourceCode.scripts;
    const resultScript = ctx.isTypeScript()
        ? parseTypeScriptInSvelte(scripts.getCurrentVirtualCodeInfo(), scripts.attrs, parserOptions, { slots: ctx.slots, svelteParseContext })
        : parseScriptInSvelte(scripts.getCurrentVirtualCode(), scripts.attrs, parserOptions);
    ctx.scriptLet.restore(resultScript);
    ctx.tokens.push(...resultScript.ast.tokens);
    ctx.comments.push(...resultScript.ast.comments);
    sortNodes(ctx.comments);
    sortNodes(ctx.tokens);
    extractTokens(ctx);
    analyzeStoreScope(resultScript.scopeManager, svelteParseContext);
    analyzeReactiveScope(resultScript.scopeManager);
    analyzeStoreScope(resultScript.scopeManager, svelteParseContext); // for reactive vars
    analyzeSnippetsScope(ctx.snippets, resultScript.scopeManager);
    // Add $$xxx variable
    addGlobalVariables(resultScript.scopeManager, getGlobalsForSvelte(svelteParseContext));
    const ast = resultTemplate.ast;
    const statements = [...resultScript.ast.body];
    ast.sourceType = resultScript.ast.sourceType;
    const scriptElements = ast.body.filter((b) => b.type === "SvelteScriptElement");
    for (let index = 0; index < scriptElements.length; index++) {
        const body = scriptElements[index];
        let statement = statements[0];
        while (statement &&
            body.range[0] <= statement.range[0] &&
            (statement.range[1] <= body.range[1] ||
                index === scriptElements.length - 1)) {
            statement.parent = body;
            body.body.push(statement);
            statements.shift();
            statement = statements[0];
        }
        if (!body.startTag.attributes.some((attr) => attr.type === "SvelteAttribute" &&
            attr.key.name === "context" &&
            attr.value.length === 1 &&
            attr.value[0].type === "SvelteLiteral" &&
            attr.value[0].value === "module")) {
            analyzePropsScope(body, resultScript.scopeManager, svelteParseContext);
        }
    }
    if (statements.length) {
        throw new ParseError("The script is unterminated", statements[0].range[1], ctx);
    }
    const styleElement = ast.body.find((b) => b.type === "SvelteStyleElement");
    let styleContext = null;
    const selectorASTs = new Map();
    resultScript.ast = ast;
    resultScript.services = Object.assign(resultScript.services || {}, {
        isSvelte: true,
        isSvelteScript: false,
        getSvelteHtmlAst() {
            return getFragmentFromRoot(resultTemplate.svelteAst);
        },
        getStyleContext() {
            if (styleContext === null) {
                styleContext = parseStyleContext(styleElement, ctx);
            }
            return styleContext;
        },
        getStyleSelectorAST(rule) {
            const cached = selectorASTs.get(rule);
            if (cached !== undefined) {
                return cached;
            }
            const ast = parseSelector(rule);
            selectorASTs.set(rule, ast);
            return ast;
        },
        styleNodeLoc,
        styleNodeRange,
        styleSelectorNodeLoc,
        svelteParseContext: {
            ...svelteParseContext,
            // The compiler decides if runes mode is used after parsing.
            runes: svelteParseContext.runes ?? hasRunesSymbol(resultScript.ast),
        },
    });
    resultScript.visitorKeys = Object.assign({}, KEYS, resultScript.visitorKeys);
    return resultScript;
}
/**
 * Parse source code as script
 */
function parseAsScript(code, parserOptions, svelteParseContext) {
    const lang = parserOptions.filePath?.split(".").pop();
    const resultScript = isTypeScript(parserOptions, lang)
        ? parseTypeScript(code, { lang }, parserOptions, svelteParseContext)
        : parseScript(code, { lang }, parserOptions);
    // Add $$xxx variable
    addGlobalVariables(resultScript.scopeManager, getGlobalsForSvelteScript(svelteParseContext));
    resultScript.services = Object.assign(resultScript.services || {}, {
        isSvelte: false,
        isSvelteScript: true,
        svelteParseContext,
    });
    resultScript.visitorKeys = Object.assign({}, KEYS, resultScript.visitorKeys);
    return resultScript;
}
function addGlobalVariables(scopeManager, globals) {
    const globalScope = scopeManager.globalScope;
    for (const globalName of globals) {
        if (globalScope.set.has(globalName))
            continue;
        const variable = new Variable();
        variable.name = globalName;
        variable.scope = globalScope;
        globalScope.variables.push(variable);
        globalScope.set.set(globalName, variable);
        globalScope.through = globalScope.through.filter((reference) => {
            if (reference.identifier.name === globalName) {
                // Links the variable and the reference.
                // And this reference is removed from `Scope#through`.
                reference.resolved = variable;
                addReference(variable.references, reference);
                return false;
            }
            return true;
        });
    }
}
/** Extract tokens */
function extractTokens(ctx) {
    const useRanges = sortNodes([...ctx.tokens, ...ctx.comments]).map((t) => t.range);
    let range = useRanges.shift();
    for (let index = 0; index < ctx.sourceCode.template.length; index++) {
        while (range && range[1] <= index) {
            range = useRanges.shift();
        }
        if (range && range[0] <= index) {
            index = range[1] - 1;
            continue;
        }
        const c = ctx.sourceCode.template[index];
        if (!c.trim()) {
            continue;
        }
        if (isPunctuator(c)) {
            ctx.addToken("Punctuator", { start: index, end: index + 1 });
        }
        else {
            // unknown
            // It is may be a bug.
            ctx.addToken("Identifier", { start: index, end: index + 1 });
        }
    }
    sortNodes(ctx.comments);
    sortNodes(ctx.tokens);
    /**
     * Checks if the given char is punctuator
     */
    function isPunctuator(c) {
        return /^[^\w$]$/iu.test(c);
    }
}
