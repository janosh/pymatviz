import { loadModule } from '../../../utils/load-module.js';
/**
 * Transpile with sass
 */
export function transform(node, text, context, type) {
    const sass = loadSass(context);
    if (!sass) {
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
        const output = sass.compileString(code, {
            sourceMap: true,
            syntax: type === 'sass' ? 'indented' : undefined
        });
        if (!output) {
            return null;
        }
        return {
            inputRange,
            output: output.css,
            mappings: output.sourceMap.mappings
        };
    }
    catch {
        return null;
    }
}
/**
 * Load sass
 */
function loadSass(context) {
    return loadModule(context, 'sass');
}
