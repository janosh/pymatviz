import postcss from 'postcss';
import postcssLoadConfig from 'postcss-load-config';
/**
 * Transform with postcss
 */
export function transform(node, text, context) {
    const postcssConfig = context.settings?.svelte?.compileOptions?.postcss;
    if (postcssConfig === false) {
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
    const filename = `${context.filename}.css`;
    try {
        const configFilePath = postcssConfig?.configFilePath;
        const config = postcssLoadConfig.sync({
            cwd: context.cwd ?? process.cwd(),
            from: filename
        }, typeof configFilePath === 'string' ? configFilePath : undefined);
        const result = postcss(config.plugins).process(code, {
            ...config.options,
            map: {
                inline: false
            }
        });
        return {
            inputRange,
            output: result.content,
            mappings: result.map.toJSON().mappings
        };
    }
    catch {
        // console.log(e)
        return null;
    }
}
