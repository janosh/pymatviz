import { loadModule } from '../../../utils/load-module.js';
/**
 * Transpile with less
 */
export function transform(node, text, context) {
    const less = loadLess(context);
    if (!less) {
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
    const filename = `${context.filename}.less`;
    try {
        let output;
        less.render(code, {
            sourceMap: {},
            syncImport: true,
            filename,
            lint: false
        }, (_error, result) => {
            output = result;
        });
        if (!output) {
            return null;
        }
        return {
            inputRange,
            output: output.css,
            mappings: JSON.parse(output.map).mappings
        };
    }
    catch {
        return null;
    }
}
/**
 * Load less
 */
function loadLess(context) {
    return loadModule(context, 'less');
}
