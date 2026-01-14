import { loadModule } from '../../../utils/load-module.js';
/**
 * Transpile with babel
 */
export function transform(node, text, context) {
    const babel = loadBabel(context);
    if (!babel) {
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
        const output = babel.transformSync(code, {
            sourceType: 'module',
            sourceMaps: true,
            minified: false,
            ast: false,
            code: true,
            cwd: context.cwd ?? process.cwd()
        });
        if (!output) {
            return null;
        }
        return {
            inputRange,
            output: output.code,
            mappings: output.map.mappings
        };
    }
    catch {
        return null;
    }
}
/** Check if project has Babel. */
export function hasBabel(context) {
    return Boolean(loadBabel(context));
}
/**
 * Load babel
 */
function loadBabel(context) {
    return loadModule(context, '@babel/core');
}
