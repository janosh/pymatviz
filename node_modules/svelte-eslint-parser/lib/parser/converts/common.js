/** indexOf */
export function indexOf(str, search, start, end) {
    const endIndex = end ?? str.length;
    for (let index = start; index < endIndex; index++) {
        const c = str[index];
        if (search(c, index)) {
            return index;
        }
    }
    return -1;
}
/** lastIndexOf */
export function lastIndexOf(str, search, end) {
    for (let index = end; index >= 0; index--) {
        const c = str[index];
        if (search(c, index)) {
            return index;
        }
    }
    return -1;
}
/** Get node with location */
export function getWithLoc(node) {
    return node;
}
