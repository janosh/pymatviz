import { convertAwaitBlock, convertEachBlock, convertIfBlock, convertKeyBlock, convertSnippetBlock, } from "./block.js";
import { getWithLoc, indexOf } from "./common.js";
import { convertMustacheTag, convertDebugTag, convertRawMustacheTag, } from "./mustache.js";
import { convertText } from "./text.js";
import { convertAttributes } from "./attr.js";
import { convertConstTag } from "./const.js";
import { sortNodes } from "../sort.js";
import { ParseError } from "../../index.js";
import { convertRenderTag } from "./render.js";
import { getChildren, getFragment } from "../compat.js";
/** Convert for Fragment or Element or ... */
export function* convertChildren(fragment, parent, ctx) {
    const children = getChildren(fragment);
    if (!children)
        return;
    for (const child of children) {
        if (child.type === "Comment") {
            yield convertComment(child, parent, ctx);
            continue;
        }
        if (child.type === "Text") {
            if (!child.data && child.start === child.end) {
                continue;
            }
            yield convertText(child, parent, ctx);
            continue;
        }
        if (child.type === "RegularElement") {
            yield convertHTMLElement(child, parent, ctx);
            continue;
        }
        if (child.type === "Element") {
            if (child.name.includes(":")) {
                yield convertSpecialElement(child, parent, ctx);
            }
            else {
                yield convertHTMLElement(child, parent, ctx);
            }
            continue;
        }
        if (child.type === "Component") {
            yield convertComponentElement(child, parent, ctx);
            continue;
        }
        if (child.type === "InlineComponent") {
            if (child.name.includes(":")) {
                yield convertSpecialElement(child, parent, ctx);
            }
            else {
                yield convertComponentElement(child, parent, ctx);
            }
            continue;
        }
        if (child.type === "SvelteComponent" ||
            child.type === "SvelteElement" ||
            child.type === "SvelteSelf") {
            yield convertSpecialElement(child, parent, ctx);
            continue;
        }
        if (child.type === "SlotElement" || child.type === "Slot") {
            yield convertSlotElement(child, parent, ctx);
            continue;
        }
        if (child.type === "ExpressionTag" || child.type === "MustacheTag") {
            yield convertMustacheTag(child, parent, null, ctx);
            continue;
        }
        if (child.type === "HtmlTag" || child.type === "RawMustacheTag") {
            yield convertRawMustacheTag(child, parent, ctx);
            continue;
        }
        if (child.type === "IfBlock") {
            // {#if expr} {/if}
            yield convertIfBlock(child, parent, ctx);
            continue;
        }
        if (child.type === "EachBlock") {
            // {#each expr as item, index (key)} {/each}
            yield convertEachBlock(child, parent, ctx);
            continue;
        }
        if (child.type === "AwaitBlock") {
            // {#await promise} {:then number} {:catch error} {/await}
            yield convertAwaitBlock(child, parent, ctx);
            continue;
        }
        if (child.type === "KeyBlock") {
            // {#key expression}...{/key}
            yield convertKeyBlock(child, parent, ctx);
            continue;
        }
        if (child.type === "SnippetBlock") {
            // {#snippet x(args)}...{/snippet}
            yield convertSnippetBlock(child, parent, ctx);
            continue;
        }
        if (child.type === "SvelteWindow" || child.type === "Window") {
            yield convertWindowElement(child, parent, ctx);
            continue;
        }
        if (child.type === "SvelteBody" || child.type === "Body") {
            yield convertBodyElement(child, parent, ctx);
            continue;
        }
        if (child.type === "SvelteHead" || child.type === "Head") {
            yield convertHeadElement(child, parent, ctx);
            continue;
        }
        if (child.type === "TitleElement" || child.type === "Title") {
            yield convertTitleElement(child, parent, ctx);
            continue;
        }
        if (child.type === "SvelteOptions" || child.type === "Options") {
            yield convertOptionsElement(child, parent, ctx);
            continue;
        }
        if (child.type === "SvelteFragment" || child.type === "SlotTemplate") {
            yield convertSlotTemplateElement(child, parent, ctx);
            continue;
        }
        if (child.type === "DebugTag") {
            yield convertDebugTag(child, parent, ctx);
            continue;
        }
        if (child.type === "ConstTag") {
            yield convertConstTag(child, parent, ctx);
            continue;
        }
        if (child.type === "RenderTag") {
            yield convertRenderTag(child, parent, ctx);
            continue;
        }
        if (child.type === "SvelteDocument" || child.type === "Document") {
            yield convertDocumentElement(child, parent, ctx);
            continue;
        }
        if (child.type === "SvelteBoundary") {
            yield convertSvelteBoundaryElement(child, parent, ctx);
            continue;
        }
        throw new Error(`Unknown type:${child.type}`);
    }
}
/** Extract `let:` directives. */
function extractLetDirectives(fragment) {
    const letDirectives = [];
    const attributes = [];
    for (const attr of fragment.attributes) {
        if (attr.type === "LetDirective" || attr.type === "Let") {
            letDirectives.push(attr);
        }
        else {
            attributes.push(attr);
        }
    }
    return { letDirectives, attributes };
}
/** Check if children needs a scope. */
function needScopeByChildren(fragment) {
    const children = getChildren(fragment);
    if (!children)
        return false;
    for (const child of children) {
        if (child.type === "ConstTag") {
            return true;
        }
        if (child.type === "SnippetBlock") {
            return true;
        }
    }
    return false;
}
/** Convert for HTML Comment */
function convertComment(node, parent, ctx) {
    const comment = {
        type: "SvelteHTMLComment",
        value: node.data,
        parent,
        ...ctx.getConvertLocation(node),
    };
    ctx.addToken("HTMLComment", node);
    return comment;
}
/** Convert for HTMLElement */
function convertHTMLElement(node, parent, ctx) {
    const locs = ctx.getConvertLocation(node);
    const element = {
        type: "SvelteElement",
        kind: "html",
        name: null,
        startTag: {
            type: "SvelteStartTag",
            attributes: [],
            selfClosing: false,
            parent: null,
            range: [locs.range[0], null],
            loc: {
                start: {
                    line: locs.loc.start.line,
                    column: locs.loc.start.column,
                },
                end: null,
            },
        },
        children: [],
        endTag: null,
        parent,
        ...locs,
    };
    ctx.elements.set(element, node);
    element.startTag.parent = element;
    const elementName = node.name;
    const { letDirectives, attributes } = extractLetDirectives(node);
    const letParams = [];
    if (letDirectives.length) {
        ctx.letDirCollections.beginExtract();
        element.startTag.attributes.push(...convertAttributes(letDirectives, element.startTag, ctx));
        letParams.push(...ctx.letDirCollections.extract().getLetParams());
    }
    const fragment = getFragment(node);
    if (!letParams.length && !needScopeByChildren(fragment)) {
        element.startTag.attributes.push(...convertAttributes(attributes, element.startTag, ctx));
        element.children.push(...convertChildren(fragment, element, ctx));
    }
    else {
        ctx.scriptLet.nestBlock(element, letParams);
        element.startTag.attributes.push(...convertAttributes(attributes, element.startTag, ctx));
        sortNodes(element.startTag.attributes);
        element.children.push(...convertChildren(fragment, element, ctx));
        ctx.scriptLet.closeScope();
    }
    extractElementTags(element, ctx, {
        buildNameNode: (openTokenRange) => {
            ctx.addToken("HTMLIdentifier", openTokenRange);
            const name = {
                type: "SvelteName",
                name: elementName,
                parent: element,
                ...ctx.getConvertLocation(openTokenRange),
            };
            return name;
        },
    });
    if (element.name.name === "script" ||
        element.name.name === "style" ||
        (element.name.name === "template" && ctx.findBlock(element))) {
        // Restore the block-like element.
        for (const child of element.children) {
            if (child.type === "SvelteText") {
                child.value = ctx.code.slice(...child.range);
            }
        }
        if (element.name.name === "script") {
            ctx.stripScriptCode(element.startTag.range[1], element.endTag?.range[0] ?? element.range[1]);
        }
    }
    if (element.startTag.selfClosing && element.name.name.endsWith("-")) {
        // Restore the self-closing block.
        const selfClosingBlock = /^[a-z]-+$/iu.test(element.name.name) &&
            ctx.findSelfClosingBlock(element);
        if (selfClosingBlock) {
            element.name.name = selfClosingBlock.originalTag;
        }
    }
    return element;
}
/** Convert for Special element. e.g. <svelte:self> */
function convertSpecialElement(node, parent, ctx) {
    const locs = ctx.getConvertLocation(node);
    const element = {
        type: "SvelteElement",
        kind: "special",
        name: null,
        startTag: {
            type: "SvelteStartTag",
            attributes: [],
            selfClosing: false,
            parent: null,
            range: [locs.range[0], null],
            loc: {
                start: {
                    line: locs.loc.start.line,
                    column: locs.loc.start.column,
                },
                end: null,
            },
        },
        children: [],
        endTag: null,
        parent,
        ...locs,
    };
    ctx.elements.set(element, node);
    element.startTag.parent = element;
    const elementName = node.name;
    const { letDirectives, attributes } = extractLetDirectives(node);
    const letParams = [];
    if (letDirectives.length) {
        ctx.letDirCollections.beginExtract();
        element.startTag.attributes.push(...convertAttributes(letDirectives, element.startTag, ctx));
        letParams.push(...ctx.letDirCollections.extract().getLetParams());
    }
    const fragment = getFragment(node);
    if (!letParams.length && !needScopeByChildren(fragment)) {
        element.startTag.attributes.push(...convertAttributes(attributes, element.startTag, ctx));
        element.children.push(...convertChildren(fragment, element, ctx));
    }
    else {
        ctx.scriptLet.nestBlock(element, letParams);
        element.startTag.attributes.push(...convertAttributes(attributes, element.startTag, ctx));
        sortNodes(element.startTag.attributes);
        element.children.push(...convertChildren(fragment, element, ctx));
        ctx.scriptLet.closeScope();
    }
    const thisExpression = (node.type === "SvelteComponent" && node.expression) ||
        (node.type === "SvelteElement" && node.tag) ||
        (node.type === "InlineComponent" &&
            elementName === "svelte:component" &&
            node.expression) ||
        (node.type === "Element" && elementName === "svelte:element" && node.tag);
    if (thisExpression) {
        processThisAttribute(node, thisExpression, element, ctx);
    }
    extractElementTags(element, ctx, {
        buildNameNode: (openTokenRange) => {
            ctx.addToken("HTMLIdentifier", openTokenRange);
            const name = {
                type: "SvelteName",
                name: elementName,
                parent: element,
                ...ctx.getConvertLocation(openTokenRange),
            };
            return name;
        },
    });
    return element;
}
/** process `this=` */
function processThisAttribute(node, thisValue, element, ctx) {
    const startIndex = findStartIndexOfThis(node, ctx);
    const eqIndex = ctx.code.indexOf("=", startIndex + 4 /* t,h,i,s */);
    let thisNode;
    if (typeof thisValue === "string") {
        // Svelte v4
        // this="..."
        thisNode = createSvelteAttribute(startIndex, eqIndex, thisValue);
    }
    else {
        // this={...}
        const valueStartIndex = indexOf(ctx.code, (c) => Boolean(c.trim()), eqIndex + 1);
        if (thisValue.type === "Literal" &&
            typeof thisValue.value === "string" &&
            ctx.code[valueStartIndex] !== "{") {
            thisNode = createSvelteAttribute(startIndex, eqIndex, thisValue.value);
        }
        else {
            thisNode = createSvelteSpecialDirective(startIndex, eqIndex, thisValue);
        }
    }
    const targetIndex = element.startTag.attributes.findIndex((attr) => thisNode.range[1] <= attr.range[0]);
    if (targetIndex === -1) {
        element.startTag.attributes.push(thisNode);
    }
    else {
        element.startTag.attributes.splice(targetIndex, 0, thisNode);
    }
    /** Create SvelteAttribute */
    function createSvelteAttribute(startIndex, eqIndex, thisValue) {
        const valueStartIndex = indexOf(ctx.code, (c) => Boolean(c.trim()), eqIndex + 1);
        const quote = ctx.code.startsWith(thisValue, valueStartIndex)
            ? null
            : ctx.code[valueStartIndex];
        const literalStartIndex = quote
            ? valueStartIndex + quote.length
            : valueStartIndex;
        const literalEndIndex = literalStartIndex + thisValue.length;
        const endIndex = quote ? literalEndIndex + quote.length : literalEndIndex;
        const thisAttr = {
            type: "SvelteAttribute",
            key: null,
            boolean: false,
            value: [],
            parent: element.startTag,
            ...ctx.getConvertLocation({ start: startIndex, end: endIndex }),
        };
        thisAttr.key = {
            type: "SvelteName",
            name: "this",
            parent: thisAttr,
            ...ctx.getConvertLocation({ start: startIndex, end: eqIndex }),
        };
        thisAttr.value.push({
            type: "SvelteLiteral",
            value: thisValue,
            parent: thisAttr,
            ...ctx.getConvertLocation({
                start: literalStartIndex,
                end: literalEndIndex,
            }),
        });
        // this
        ctx.addToken("HTMLIdentifier", {
            start: startIndex,
            end: startIndex + 4,
        });
        // =
        ctx.addToken("Punctuator", {
            start: eqIndex,
            end: eqIndex + 1,
        });
        if (quote) {
            // "
            ctx.addToken("Punctuator", {
                start: valueStartIndex,
                end: literalStartIndex,
            });
        }
        ctx.addToken("HTMLText", {
            start: literalStartIndex,
            end: literalEndIndex,
        });
        if (quote) {
            // "
            ctx.addToken("Punctuator", {
                start: literalEndIndex,
                end: endIndex,
            });
        }
        return thisAttr;
    }
    /** Create SvelteSpecialDirective */
    function createSvelteSpecialDirective(startIndex, eqIndex, expression) {
        const closeIndex = ctx.code.indexOf("}", getWithLoc(expression).end);
        const endIndex = indexOf(ctx.code, (c) => c === ">" || !c.trim(), closeIndex);
        const thisDir = {
            type: "SvelteSpecialDirective",
            kind: "this",
            key: null,
            expression: null,
            parent: element.startTag,
            ...ctx.getConvertLocation({ start: startIndex, end: endIndex }),
        };
        thisDir.key = {
            type: "SvelteSpecialDirectiveKey",
            parent: thisDir,
            ...ctx.getConvertLocation({ start: startIndex, end: eqIndex }),
        };
        // this
        ctx.addToken("HTMLIdentifier", {
            start: startIndex,
            end: startIndex + 4,
        });
        // =
        ctx.addToken("Punctuator", {
            start: eqIndex,
            end: eqIndex + 1,
        });
        ctx.scriptLet.addExpression(expression, thisDir, null, (es) => {
            thisDir.expression = es;
        });
        return thisDir;
    }
}
/** Find the start index of `this` */
function findStartIndexOfThis(node, ctx) {
    // Get the end index of `svelte:element`
    const startIndex = ctx.code.indexOf(node.name, node.start) + node.name.length;
    const sortedAttrs = [...node.attributes].sort((a, b) => a.start - b.start);
    // Find the start index of `this` from the end index of `svelte:element`.
    // However, it only seeks to the start index of the first attribute (or the end index of element node).
    let thisIndex = indexOf(ctx.code, (_c, index) => ctx.code.startsWith("this", index), startIndex, sortedAttrs[0]?.start ?? node.end);
    while (thisIndex < 0) {
        if (sortedAttrs.length === 0)
            throw new ParseError("Cannot resolved `this` attribute.", thisIndex, ctx);
        // Step3: Find the start index of `this` from the end index of attribute.
        // However, it only seeks to the start index of the first attribute (or the end index of element node).
        const nextStartIndex = sortedAttrs.shift().end;
        thisIndex = indexOf(ctx.code, (_c, index) => ctx.code.startsWith("this", index), nextStartIndex, sortedAttrs[0]?.start ?? node.end);
    }
    return thisIndex;
}
/** Convert for ComponentElement */
function convertComponentElement(node, parent, ctx) {
    const locs = ctx.getConvertLocation(node);
    const element = {
        type: "SvelteElement",
        kind: "component",
        name: null,
        startTag: {
            type: "SvelteStartTag",
            attributes: [],
            selfClosing: false,
            parent: null,
            range: [locs.range[0], null],
            loc: {
                start: {
                    line: locs.loc.start.line,
                    column: locs.loc.start.column,
                },
                end: null,
            },
        },
        children: [],
        endTag: null,
        parent,
        ...locs,
    };
    ctx.elements.set(element, node);
    element.startTag.parent = element;
    const elementName = node.name;
    const { letDirectives, attributes } = extractLetDirectives(node);
    const letParams = [];
    if (letDirectives.length) {
        ctx.letDirCollections.beginExtract();
        element.startTag.attributes.push(...convertAttributes(letDirectives, element.startTag, ctx));
        letParams.push(...ctx.letDirCollections.extract().getLetParams());
    }
    const fragment = getFragment(node);
    if (!letParams.length && !needScopeByChildren(fragment)) {
        element.startTag.attributes.push(...convertAttributes(attributes, element.startTag, ctx));
        element.children.push(...convertChildren(fragment, element, ctx));
    }
    else {
        ctx.scriptLet.nestBlock(element, letParams);
        element.startTag.attributes.push(...convertAttributes(attributes, element.startTag, ctx));
        sortNodes(element.startTag.attributes);
        element.children.push(...convertChildren(fragment, element, ctx));
        ctx.scriptLet.closeScope();
    }
    extractElementTags(element, ctx, {
        buildNameNode: (openTokenRange) => {
            const chains = elementName.split(".");
            const id = chains.shift();
            const idRange = {
                start: openTokenRange.start,
                end: openTokenRange.start + id.length,
            };
            // ctx.addToken("Identifier", idRange)
            const identifier = {
                type: "Identifier",
                name: id,
                // @ts-expect-error -- ignore
                parent: element,
                ...ctx.getConvertLocation(idRange),
            };
            let object = identifier;
            // eslint-disable-next-line func-style -- var
            let esCallback = (es) => {
                element.name = es;
            };
            let start = idRange.end + 1;
            for (const name of chains) {
                const range = { start, end: start + name.length };
                ctx.addToken("HTMLIdentifier", range);
                const mem = {
                    type: "SvelteMemberExpressionName",
                    object,
                    property: {
                        type: "SvelteName",
                        name,
                        parent: null,
                        ...ctx.getConvertLocation(range),
                    },
                    parent: element,
                    ...ctx.getConvertLocation({
                        start: openTokenRange.start,
                        end: range.end,
                    }),
                };
                mem.property.parent = mem;
                object.parent = mem;
                object = mem;
                start = range.end + 1;
                if (mem.object === identifier) {
                    esCallback = (es) => {
                        mem.object = es;
                    };
                }
            }
            ctx.scriptLet.addExpression(identifier, identifier.parent, null, esCallback);
            return object;
        },
    });
    return element;
}
/** Convert for Slot */
function convertSlotElement(node, parent, ctx) {
    // Slot translates to SvelteHTMLElement.
    const element = convertHTMLElement(node, parent, ctx);
    ctx.slots.add(element);
    return element;
}
/** Convert for window element. e.g. <svelte:window> */
function convertWindowElement(node, parent, ctx) {
    return convertSpecialElement(node, parent, ctx);
}
/** Convert for document element. e.g. <svelte:document> */
function convertDocumentElement(node, parent, ctx) {
    return convertSpecialElement(node, parent, ctx);
}
/** Convert for body element. e.g. <svelte:body> */
function convertBodyElement(node, parent, ctx) {
    return convertSpecialElement(node, parent, ctx);
}
/** Convert for head element. e.g. <svelte:head> */
function convertHeadElement(node, parent, ctx) {
    return convertSpecialElement(node, parent, ctx);
}
/** Convert for title element. e.g. <title> */
function convertTitleElement(node, parent, ctx) {
    return convertHTMLElement(node, parent, ctx);
}
/** Convert for options element. e.g. <svelte:options> */
function convertOptionsElement(node, parent, ctx) {
    return convertSpecialElement(node, parent, ctx);
}
/** Convert for <svelte:fragment> element. */
function convertSlotTemplateElement(node, parent, ctx) {
    return convertSpecialElement(node, parent, ctx);
}
/** Convert for <svelte:boundary> element. */
function convertSvelteBoundaryElement(node, parent, ctx) {
    return convertSpecialElement(node, parent, ctx);
}
/** Extract element tag and tokens */
export function extractElementTags(element, ctx, options) {
    const startTagNameEnd = indexOf(ctx.code, (c) => c === "/" || c === ">" || !c.trim(), element.range[0] + 1);
    const openTokenRange = {
        start: element.range[0] + 1,
        end: startTagNameEnd,
    };
    element.name = options.buildNameNode(openTokenRange);
    const startTagEnd = ctx.code.indexOf(">", element.startTag.attributes[element.startTag.attributes.length - 1]
        ?.range[1] ?? openTokenRange.end) + 1;
    element.startTag.range[1] = startTagEnd;
    element.startTag.loc.end = ctx.getLocFromIndex(startTagEnd);
    if (ctx.code[element.range[1] - 1] !== ">") {
        // Have not end tag
        return;
    }
    if (ctx.code[element.range[1] - 2] === "/") {
        // self close
        element.startTag.selfClosing = true;
        return;
    }
    const endTagOpen = ctx.code.lastIndexOf("<", element.range[1] - 1);
    if (endTagOpen <= startTagEnd - 1) {
        // void element
        return;
    }
    const endTagNameStart = endTagOpen + 2;
    const endTagNameEnd = indexOf(ctx.code, (c) => c === ">" || !c.trim(), endTagNameStart);
    const endTagClose = ctx.code.indexOf(">", endTagNameEnd);
    element.endTag = {
        type: "SvelteEndTag",
        parent: element,
        ...ctx.getConvertLocation({ start: endTagOpen, end: endTagClose + 1 }),
    };
    ctx.addToken("HTMLIdentifier", {
        start: endTagNameStart,
        end: endTagNameEnd,
    });
}
