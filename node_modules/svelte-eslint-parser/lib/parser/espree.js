import Module from "module";
import path from "path";
let espreeCache = null;
/** Checks if given path is linter path */
function isLinterPath(p) {
    return (
    // ESLint 6 and above
    p.includes(`eslint${path.sep}lib${path.sep}linter${path.sep}linter.js`) ||
        // ESLint 5
        p.includes(`eslint${path.sep}lib${path.sep}linter.js`));
}
/**
 * Load `espree` from the loaded ESLint.
 * If the loaded ESLint was not found, just returns `require("espree")`.
 */
export function getEspree() {
    if (!espreeCache) {
        // Lookup the loaded eslint
        const req = Module.createRequire(import.meta.url);
        const linterPath = Object.keys(req.cache || {}).find(isLinterPath);
        if (linterPath) {
            try {
                espreeCache = Module.createRequire(linterPath)("espree");
            }
            catch {
                // ignore
            }
        }
        if (!espreeCache) {
            try {
                return req("espree");
            }
            catch {
                // ignore
            }
        }
        if (!espreeCache) {
            if (typeof require === "function")
                // eslint-disable-next-line @typescript-eslint/no-require-imports -- ignore
                espreeCache = require("espree");
        }
    }
    return espreeCache;
}
