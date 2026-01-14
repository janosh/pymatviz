export var ElementOccurenceCount;
(function (ElementOccurenceCount) {
    ElementOccurenceCount[ElementOccurenceCount["ZeroOrOne"] = 0] = "ZeroOrOne";
    ElementOccurenceCount[ElementOccurenceCount["One"] = 1] = "One";
    ElementOccurenceCount[ElementOccurenceCount["ZeroToInf"] = 2] = "ZeroToInf";
})(ElementOccurenceCount || (ElementOccurenceCount = {}));
function multiplyCounts(left, right) {
    if (left === ElementOccurenceCount.One) {
        return right;
    }
    if (right === ElementOccurenceCount.One) {
        return left;
    }
    if (left === right) {
        return left;
    }
    return ElementOccurenceCount.ZeroToInf;
}
function childElementOccurenceCount(parent) {
    if (parent === null) {
        return ElementOccurenceCount.One;
    }
    if ([
        'SvelteIfBlock',
        'SvelteElseBlock',
        'SvelteAwaitBlock',
        'SvelteAwaitPendingBlock',
        'SvelteAwaitThenBlock',
        'SvelteAwaitCatchBlock'
    ].includes(parent.type)) {
        return ElementOccurenceCount.ZeroOrOne;
    }
    if (['SvelteEachBlock', 'SvelteSnippetBlock'].includes(parent.type) ||
        (parent.type === 'SvelteElement' && parent.kind === 'component')) {
        return ElementOccurenceCount.ZeroToInf;
    }
    return ElementOccurenceCount.One;
}
export function elementOccurrenceCount(element) {
    const parentCount = element.parent !== null ? elementOccurrenceCount(element.parent) : ElementOccurenceCount.One;
    const parentChildCount = childElementOccurenceCount(element.parent);
    return multiplyCounts(parentCount, parentChildCount);
}
