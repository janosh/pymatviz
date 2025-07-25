/**
 * Sort tokens
 */
export function sortNodes(tokens) {
    if (!tokens) {
        return [];
    }
    return tokens.sort((a, b) => {
        if (a.range[0] !== b.range[0]) {
            return a.range[0] - b.range[0];
        }
        return a.range[1] - b.range[1];
    });
}
