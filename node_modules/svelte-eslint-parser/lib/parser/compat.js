// Root
export function getFragmentFromRoot(svelteAst) {
    return (svelteAst.fragment ?? svelteAst.html);
}
export function getInstanceFromRoot(svelteAst) {
    return svelteAst.instance;
}
export function getModuleFromRoot(svelteAst) {
    return svelteAst.module;
}
export function getOptionsFromRoot(svelteAst) {
    const root = svelteAst;
    if (root.options) {
        return {
            type: "SvelteOptions",
            name: "svelte:options",
            attributes: root.options.attributes,
            fragment: {
                type: "Fragment",
                nodes: [],
            },
            start: root.options.start,
            end: root.options.end,
        };
    }
    return null;
}
export function getChildren(fragment) {
    return (fragment.nodes ??
        fragment.children);
}
export function trimChildren(children) {
    if (!startsWithWhitespace(children[0]) &&
        !endsWithWhitespace(children[children.length - 1])) {
        return children;
    }
    const nodes = [...children];
    while (isWhitespace(nodes[0])) {
        nodes.shift();
    }
    const first = nodes[0];
    if (startsWithWhitespace(first)) {
        nodes[0] = { ...first, data: first.data.trimStart() };
    }
    while (isWhitespace(nodes[nodes.length - 1])) {
        nodes.pop();
    }
    const last = nodes[nodes.length - 1];
    if (endsWithWhitespace(last)) {
        nodes[nodes.length - 1] = { ...last, data: last.data.trimEnd() };
    }
    return nodes;
    function startsWithWhitespace(child) {
        if (!child) {
            return false;
        }
        return child.type === "Text" && child.data.trimStart() !== child.data;
    }
    function endsWithWhitespace(child) {
        if (!child) {
            return false;
        }
        return child.type === "Text" && child.data.trimEnd() !== child.data;
    }
    function isWhitespace(child) {
        if (!child) {
            return false;
        }
        return child.type === "Text" && child.data.trim() === "";
    }
}
export function getFragment(element) {
    if (element.fragment) {
        return element.fragment;
    }
    return element;
}
export function getModifiers(node) {
    return node.modifiers ?? [];
}
// IfBlock
export function getTestFromIfBlock(block) {
    return (block.expression ?? block.test);
}
export function getConsequentFromIfBlock(block) {
    return block.consequent ?? block;
}
export function getAlternateFromIfBlock(block) {
    if (block.alternate) {
        return block.alternate;
    }
    return block.else ?? null;
}
// EachBlock
export function getBodyFromEachBlock(block) {
    if (block.body) {
        return block.body;
    }
    return block;
}
export function getFallbackFromEachBlock(block) {
    if (block.fallback) {
        return block.fallback;
    }
    return block.else ?? null;
}
// AwaitBlock
export function getPendingFromAwaitBlock(block) {
    const pending = block.pending;
    if (!pending) {
        return null;
    }
    if (pending.type === "Fragment") {
        return pending;
    }
    return pending.skip ? null : pending;
}
export function getThenFromAwaitBlock(block) {
    const then = block.then;
    if (!then) {
        return null;
    }
    if (then.type === "Fragment") {
        return then;
    }
    return then.skip ? null : then;
}
export function getCatchFromAwaitBlock(block) {
    const catchFragment = block.catch;
    if (!catchFragment) {
        return null;
    }
    if (catchFragment.type === "Fragment") {
        return catchFragment;
    }
    return catchFragment.skip ? null : catchFragment;
}
// ConstTag
export function getDeclaratorFromConstTag(node) {
    return (node.declaration?.declarations?.[0] ??
        node.expression);
}
