import { loadModule } from '../../../utils/load-module.js';
/**
 * Transpile with typescript
 */
export function transform(node, text, context) {
    const ts = loadTs(context);
    if (!ts) {
        return null;
    }
    let inputRange;
    if (node.endTag) {
        inputRange = [node.startTag.range[1], node.endTag.range[0]];
    }
    else {
        inputRange = [node.startTag.range[1], node.range[1]];
    }
    const code = text.slice(...inputRange);
    try {
        const output = ts.transpileModule(code, {
            reportDiagnostics: false,
            compilerOptions: {
                target: context.sourceCode.parserServices.program?.getCompilerOptions()?.target ||
                    ts.ScriptTarget.ESNext,
                module: ts.ModuleKind.ESNext,
                importsNotUsedAsValues: ts.ImportsNotUsedAsValues.Preserve,
                preserveValueImports: true,
                verbatimModuleSyntax: true,
                sourceMap: true
            }
        });
        return {
            inputRange,
            output: output.outputText,
            mappings: JSON.parse(output.sourceMapText).mappings
        };
    }
    catch {
        return null;
    }
}
/** Check if project has TypeScript. */
export function hasTypeScript(context) {
    return Boolean(loadTs(context));
}
/**
 * Load typescript
 */
function loadTs(context) {
    return loadModule(context, 'typescript');
}
