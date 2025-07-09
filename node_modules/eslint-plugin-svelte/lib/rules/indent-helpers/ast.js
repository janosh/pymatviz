/**
 * Check whether the given token is a whitespace.
 */
export function isWhitespace(token) {
    return (token != null &&
        ((token.type === 'HTMLText' && !token.value.trim()) ||
            (token.type === 'JSXText' && !token.value.trim())));
}
/**
 * Check whether the given token is a not whitespace.
 */
export function isNotWhitespace(token) {
    return (token != null &&
        (token.type !== 'HTMLText' || Boolean(token.value.trim())) &&
        (token.type !== 'JSXText' || Boolean(token.value.trim())));
}
