import type { AST } from 'svelte-eslint-parser';
export declare enum ElementOccurenceCount {
    ZeroOrOne = 0,
    One = 1,
    ZeroToInf = 2
}
export declare function elementOccurrenceCount(element: AST.SvelteHTMLNode): ElementOccurenceCount;
