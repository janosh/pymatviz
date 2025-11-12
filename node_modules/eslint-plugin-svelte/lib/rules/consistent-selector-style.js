import { findClassesInAttribute } from '../utils/ast-utils.js';
import { extractExpressionPrefixLiteral, extractExpressionSuffixLiteral } from '../utils/expression-affixes.js';
import { createRule } from '../utils/index.js';
import { ElementOccurenceCount, elementOccurrenceCount } from '../utils/element-occurences.js';
export default createRule('consistent-selector-style', {
    meta: {
        docs: {
            description: 'enforce a consistent style for CSS selectors',
            category: 'Stylistic Issues',
            recommended: false,
            conflictWithPrettier: false
        },
        schema: [
            {
                type: 'object',
                properties: {
                    checkGlobal: {
                        type: 'boolean'
                    },
                    style: {
                        type: 'array',
                        items: {
                            enum: ['class', 'id', 'type']
                        },
                        maxItems: 3,
                        uniqueItems: true
                    }
                },
                additionalProperties: false
            }
        ],
        messages: {
            classShouldBeId: 'Selector should select by ID instead of class',
            classShouldBeType: 'Selector should select by element type instead of class',
            idShouldBeClass: 'Selector should select by class instead of ID',
            idShouldBeType: 'Selector should select by element type instead of ID',
            typeShouldBeClass: 'Selector should select by class instead of element type',
            typeShouldBeId: 'Selector should select by ID instead of element type'
        },
        type: 'suggestion'
    },
    create(context) {
        const sourceCode = context.sourceCode;
        if (!sourceCode.parserServices.isSvelte ||
            sourceCode.parserServices.getStyleSelectorAST === undefined ||
            sourceCode.parserServices.styleSelectorNodeLoc === undefined) {
            return {};
        }
        const getStyleSelectorAST = sourceCode.parserServices.getStyleSelectorAST;
        const styleSelectorNodeLoc = sourceCode.parserServices.styleSelectorNodeLoc;
        const checkGlobal = context.options[0]?.checkGlobal ?? false;
        const style = context.options[0]?.style ?? ['type', 'id', 'class'];
        const whitelistedClasses = [];
        const selections = {
            class: {
                exact: new Map(),
                affixes: new Map(),
                universalSelector: false
            },
            id: {
                exact: new Map(),
                affixes: new Map(),
                universalSelector: false
            },
            type: new Map()
        };
        /**
         * Checks selectors in a given PostCSS node
         */
        function checkSelectorsInPostCSSNode(node) {
            if (node.type === 'rule') {
                checkSelector(getStyleSelectorAST(node));
            }
            if ((node.type === 'root' ||
                (node.type === 'rule' && (node.selector !== ':global' || checkGlobal)) ||
                node.type === 'atrule') &&
                node.nodes !== undefined) {
                node.nodes.flatMap((node) => checkSelectorsInPostCSSNode(node));
            }
        }
        /**
         * Checks an individual selector
         */
        function checkSelector(node) {
            if (node.type === 'class') {
                checkClassSelector(node);
            }
            if (node.type === 'id') {
                checkIdSelector(node);
            }
            if (node.type === 'tag') {
                checkTypeSelector(node);
            }
            if ((node.type === 'pseudo' && (node.value !== ':global' || checkGlobal)) ||
                node.type === 'root' ||
                node.type === 'selector') {
                node.nodes.flatMap((node) => checkSelector(node));
            }
        }
        /**
         * Checks a class selector
         */
        function checkClassSelector(node) {
            if (selections.class.universalSelector || whitelistedClasses.includes(node.value)) {
                return;
            }
            const selection = matchSelection(selections.class, node.value);
            for (const styleValue of style) {
                if (styleValue === 'class') {
                    return;
                }
                if (styleValue === 'id' && canUseIdSelector(selection.map(([elem]) => elem))) {
                    context.report({
                        messageId: 'classShouldBeId',
                        loc: styleSelectorNodeLoc(node)
                    });
                    return;
                }
                if (styleValue === 'type' && canUseTypeSelector(selection, selections.type)) {
                    context.report({
                        messageId: 'classShouldBeType',
                        loc: styleSelectorNodeLoc(node)
                    });
                    return;
                }
            }
        }
        /**
         * Checks an ID selector
         */
        function checkIdSelector(node) {
            if (selections.id.universalSelector) {
                return;
            }
            const selection = matchSelection(selections.id, node.value);
            for (const styleValue of style) {
                if (styleValue === 'class') {
                    context.report({
                        messageId: 'idShouldBeClass',
                        loc: styleSelectorNodeLoc(node)
                    });
                    return;
                }
                if (styleValue === 'id') {
                    return;
                }
                if (styleValue === 'type' && canUseTypeSelector(selection, selections.type)) {
                    context.report({
                        messageId: 'idShouldBeType',
                        loc: styleSelectorNodeLoc(node)
                    });
                    return;
                }
            }
        }
        /**
         * Checks a type selector
         */
        function checkTypeSelector(node) {
            const selection = selections.type.get(node.value) ?? [];
            for (const styleValue of style) {
                if (styleValue === 'class') {
                    context.report({
                        messageId: 'typeShouldBeClass',
                        loc: styleSelectorNodeLoc(node)
                    });
                    return;
                }
                if (styleValue === 'id' && canUseIdSelector(selection)) {
                    context.report({
                        messageId: 'typeShouldBeId',
                        loc: styleSelectorNodeLoc(node)
                    });
                    return;
                }
                if (styleValue === 'type') {
                    return;
                }
            }
        }
        return {
            SvelteElement(node) {
                if (node.kind !== 'html') {
                    return;
                }
                addToArrayMap(selections.type, node.name.name, node);
                for (const attribute of node.startTag.attributes) {
                    if (attribute.type === 'SvelteDirective' && attribute.kind === 'Class') {
                        whitelistedClasses.push(attribute.key.name.name);
                    }
                    for (const className of findClassesInAttribute(attribute)) {
                        addToArrayMap(selections.class.exact, className, node);
                    }
                    if (attribute.type !== 'SvelteAttribute') {
                        continue;
                    }
                    for (const value of attribute.value) {
                        if (attribute.key.name === 'class' && value.type === 'SvelteMustacheTag') {
                            const prefix = extractExpressionPrefixLiteral(context, value.expression);
                            const suffix = extractExpressionSuffixLiteral(context, value.expression);
                            if (prefix === null && suffix === null) {
                                selections.class.universalSelector = true;
                            }
                            else {
                                addToArrayMap(selections.class.affixes, [prefix, suffix], node);
                            }
                        }
                        if (attribute.key.name === 'id') {
                            if (value.type === 'SvelteLiteral') {
                                addToArrayMap(selections.id.exact, value.value, node);
                            }
                            else if (value.type === 'SvelteMustacheTag') {
                                const prefix = extractExpressionPrefixLiteral(context, value.expression);
                                const suffix = extractExpressionSuffixLiteral(context, value.expression);
                                if (prefix === null && suffix === null) {
                                    selections.id.universalSelector = true;
                                }
                                else {
                                    addToArrayMap(selections.id.affixes, [prefix, suffix], node);
                                }
                            }
                        }
                    }
                }
            },
            'Program:exit'() {
                const styleContext = sourceCode.parserServices.getStyleContext();
                if (styleContext.status !== 'success' ||
                    sourceCode.parserServices.getStyleSelectorAST === undefined) {
                    return;
                }
                checkSelectorsInPostCSSNode(styleContext.sourceAst);
            }
        };
    }
});
/**
 * Helper function to add a value to a Map of arrays
 */
function addToArrayMap(map, key, value) {
    map.set(key, (map.get(key) ?? []).concat(value));
}
/**
 * Finds all nodes in selections that could be matched by key
 */
function matchSelection(selections, key) {
    const selection = (selections.exact.get(key) ?? []).map((elem) => [elem, true]);
    selections.affixes.forEach((nodes, [prefix, suffix]) => {
        if ((prefix === null || key.startsWith(prefix)) && (suffix === null || key.endsWith(suffix))) {
            selection.push(...nodes.map((elem) => [elem, false]));
        }
    });
    return selection;
}
/**
 * Checks whether a given selection could be obtained using an ID selector
 */
function canUseIdSelector(selection) {
    return (selection.length === 0 ||
        (selection.length === 1 &&
            elementOccurrenceCount(selection[0]) !== ElementOccurenceCount.ZeroToInf));
}
/**
 * Checks whether a given selection could be obtained using a type selector
 */
function canUseTypeSelector(selection, typeSelections) {
    const types = new Set(selection.map(([node]) => node.name.name));
    if (types.size > 1) {
        return false;
    }
    if (types.size < 1) {
        return true;
    }
    if (selection.some(([elem, exact]) => !exact && elementOccurrenceCount(elem) === ElementOccurenceCount.ZeroToInf)) {
        return false;
    }
    const type = [...types][0];
    const typeSelection = typeSelections.get(type);
    return (typeSelection !== undefined &&
        arrayEquals(typeSelection, selection.map(([elem]) => elem)));
}
/**
 * Compares two arrays for item equality
 */
function arrayEquals(array1, array2) {
    return array1.length === array2.length && array1.every((e) => array2.includes(e));
}
