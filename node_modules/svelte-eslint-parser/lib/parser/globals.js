const globalsForSvelte = ["$$slots", "$$props", "$$restProps"];
export const globalsForRunes = [
    "$state",
    "$derived",
    "$effect",
    "$props",
    "$bindable",
    "$inspect",
    "$host",
];
export function getGlobalsForSvelte(svelteParseContext) {
    // Process if not confirmed as non-Runes mode.
    if (svelteParseContext.runes !== false) {
        return [...globalsForSvelte, ...globalsForRunes];
    }
    return globalsForSvelte;
}
export function getGlobalsForSvelteScript(svelteParseContext) {
    // Process if not confirmed as non-Runes mode.
    if (svelteParseContext.runes !== false) {
        return globalsForRunes;
    }
    return [];
}
