/** Checks whether given object is ParserObject */
export function isParserObject(value) {
    return isEnhancedParserObject(value) || isBasicParserObject(value);
}
/** Checks whether given object is EnhancedParserObject */
export function isEnhancedParserObject(value) {
    return Boolean(value && typeof value.parseForESLint === "function");
}
/** Checks whether given object is BasicParserObject */
export function isBasicParserObject(value) {
    return Boolean(value && typeof value.parse === "function");
}
/** Checks whether given object maybe "@typescript-eslint/parser" */
export function maybeTSESLintParserObject(value) {
    return (isEnhancedParserObject(value) &&
        isBasicParserObject(value) &&
        typeof value.createProgram === "function" &&
        typeof value.clearCaches === "function" &&
        typeof value.version === "string");
}
/** Checks whether given object is "@typescript-eslint/parser" */
export function isTSESLintParserObject(value) {
    if (!isEnhancedParserObject(value))
        return false;
    try {
        const result = value.parseForESLint("", {});
        const services = result.services;
        return Boolean(services &&
            services.esTreeNodeToTSNodeMap &&
            services.tsNodeToESTreeNodeMap);
    }
    catch {
        return false;
    }
}
