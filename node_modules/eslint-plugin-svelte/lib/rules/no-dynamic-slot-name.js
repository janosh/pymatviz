import { createRule } from '../utils/index.js';
import { findVariable, getAttributeValueQuoteAndRange, getStringIfConstant } from '../utils/ast-utils.js';
export default createRule('no-dynamic-slot-name', {
    meta: {
        docs: {
            description: 'disallow dynamic slot name',
            category: 'Possible Errors',
            recommended: false
        },
        fixable: 'code',
        deprecated: true,
        schema: [],
        messages: {
            unexpected: '`<slot>` name cannot be dynamic.',
            requireValue: '`<slot>` name requires a value.'
        },
        type: 'problem',
        replacedBy: {
            note: 'Now Svelte compiler itself throws an compile error.'
        }
    },
    create(context) {
        const sourceCode = context.sourceCode;
        return {
            "SvelteElement[name.name='slot'] > SvelteStartTag.startTag > SvelteAttribute[key.name='name']"(node) {
                if (node.value.length === 0) {
                    context.report({
                        node,
                        messageId: 'requireValue'
                    });
                    return;
                }
                for (const vNode of node.value) {
                    if (vNode.type === 'SvelteMustacheTag') {
                        context.report({
                            node: vNode,
                            messageId: 'unexpected',
                            fix(fixer) {
                                const text = getStaticText(vNode.expression);
                                if (text == null) {
                                    return null;
                                }
                                if (node.value.length === 1) {
                                    const range = getAttributeValueQuoteAndRange(node, sourceCode).range;
                                    return fixer.replaceTextRange(range, `"${text}"`);
                                }
                                const range = vNode.range;
                                return fixer.replaceTextRange(range, text);
                            }
                        });
                    }
                }
            }
        };
        /**
         * Get static text from given expression
         */
        function getStaticText(node) {
            const expr = findRootExpression(node);
            return getStringIfConstant(expr);
        }
        /** Find data expression */
        function findRootExpression(node, already = new Set()) {
            if (node.type !== 'Identifier' || already.has(node)) {
                return node;
            }
            already.add(node);
            const variable = findVariable(context, node);
            if (!variable || variable.defs.length !== 1) {
                return node;
            }
            const def = variable.defs[0];
            if (def.type === 'Variable') {
                if (def.parent.kind === 'const' && def.node.init) {
                    const init = def.node.init;
                    return findRootExpression(init, already);
                }
            }
            return node;
        }
    }
});
