import { loadModule } from '../../../utils/load-module.js';
/**
 * Transpile with stylus
 */
export function transform(node, text, context) {
    const stylus = loadStylus(context);
    if (!stylus) {
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
    const filename = `${context.filename}.stylus`;
    try {
        let output;
        const style = stylus(code, {
            filename
        }).set('sourcemap', {});
        style.render((_error, code) => {
            output = code;
        });
        if (output == null) {
            return null;
        }
        return {
            inputRange,
            output,
            mappings: style.sourcemap.mappings
        };
    }
    catch {
        return null;
    }
}
/**
 * Load stylus
 */
function loadStylus(context) {
    return loadModule(context, 'stylus');
}
