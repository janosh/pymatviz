/**
 * Checks whether given property name has vender prefix
 */
export function hasVendorPrefix(prop) {
    return Boolean(getVendorPrefix(prop));
}
/**
 * Get the vender prefix from given property name
 */
export function getVendorPrefix(prop) {
    return /^-\w+-/u.exec(prop)?.[0] || '';
}
/**
 * Strip the vender prefix
 */
export function stripVendorPrefix(prop) {
    return prop.slice(getVendorPrefix(prop).length);
}
