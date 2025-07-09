import { createRule } from '../utils/index.js';
import { extractStoreReferences } from './reference-helpers/svelte-store.js';
import { findVariableForReplacement } from '../utils/ast-utils.js';
function createFixer(node, variable, name) {
    return function* fix(fixer) {
        yield fixer.replaceText(node, name);
        if (variable) {
            for (const ref of variable.references) {
                yield fixer.replaceText(ref.identifier, name);
            }
        }
    };
}
export default createRule('derived-has-same-inputs-outputs', {
    meta: {
        docs: {
            description: 'derived store should use same variable names between values and callback',
            category: 'Stylistic Issues',
            recommended: false,
            conflictWithPrettier: false
        },
        hasSuggestions: true,
        schema: [],
        messages: {
            unexpected: "The argument name should be '{{name}}'.",
            renameParam: 'Rename the parameter from {{oldName}} to {{newName}}.'
        },
        type: 'suggestion'
    },
    create(context) {
        /** check node type */
        function isIdentifierOrArrayExpression(node) {
            return ['Identifier', 'ArrayExpression'].includes(node.type);
        }
        /** check node type */
        function isFunctionExpression(node) {
            return ['ArrowFunctionExpression', 'FunctionExpression'].includes(node.type);
        }
        /**
         * Check for identifier type.
         * e.g. derived(a, ($a) => {});
         */
        function checkIdentifier(context, args, fn) {
            const fnParam = fn.params[0];
            if (fnParam.type !== 'Identifier')
                return;
            const expectedName = `$${args.name}`;
            if (expectedName !== fnParam.name) {
                const { hasConflict, variable } = findVariableForReplacement(context, fn.body, fnParam.name, expectedName);
                context.report({
                    node: fn,
                    loc: fnParam.loc,
                    messageId: 'unexpected',
                    data: { name: expectedName },
                    suggest: hasConflict
                        ? undefined
                        : [
                            {
                                messageId: 'renameParam',
                                data: { oldName: fnParam.name, newName: expectedName },
                                fix: createFixer(fnParam, variable, expectedName)
                            }
                        ]
                });
            }
        }
        /**
         * Check for array type.
         * e.g. derived([ a, b ], ([ $a, $b ]) => {})
         */
        function checkArrayExpression(context, args, fn) {
            const fnParam = fn.params[0];
            if (fnParam.type !== 'ArrayPattern')
                return;
            const argNames = args.elements.map((element) => {
                return element && element.type === 'Identifier' ? element.name : null;
            });
            fnParam.elements.forEach((element, index) => {
                const argName = argNames[index];
                if (element && element.type === 'Identifier' && argName) {
                    const expectedName = `$${argName}`;
                    if (expectedName !== element.name) {
                        const { hasConflict, variable } = findVariableForReplacement(context, fn.body, element.name, expectedName);
                        context.report({
                            node: fn,
                            loc: element.loc,
                            messageId: 'unexpected',
                            data: { name: expectedName },
                            suggest: hasConflict
                                ? undefined
                                : [
                                    {
                                        messageId: 'renameParam',
                                        data: { oldName: element.name, newName: expectedName },
                                        fix: createFixer(element, variable, expectedName)
                                    }
                                ]
                        });
                    }
                }
            });
        }
        return {
            Program() {
                for (const { node } of extractStoreReferences(context, ['derived'])) {
                    const [args, fn] = node.arguments;
                    if (!args || !isIdentifierOrArrayExpression(args))
                        continue;
                    if (!fn || !isFunctionExpression(fn))
                        continue;
                    if (!fn.params || fn.params.length === 0)
                        continue;
                    if (args.type === 'Identifier')
                        checkIdentifier(context, args, fn);
                    else
                        checkArrayExpression(context, args, fn);
                }
            }
        };
    }
});
