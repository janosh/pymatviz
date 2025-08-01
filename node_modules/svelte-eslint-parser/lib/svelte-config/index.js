import path from "path";
import fs from "fs";
import { parseConfig } from "./parser.js";
const caches = new Map();
/**
 * Resolves svelte.config.
 */
export function resolveSvelteConfigFromOption(options) {
    if (options?.svelteConfig) {
        return options.svelteConfig;
    }
    return resolveSvelteConfig(options?.filePath);
}
/**
 * Resolves `svelte.config.js`.
 * It searches the parent directories of the given file to find `svelte.config.js`,
 * and returns the static analysis result for it.
 */
function resolveSvelteConfig(filePath) {
    let cwd = filePath && fs.existsSync(filePath) ? path.dirname(filePath) : null;
    if (cwd == null) {
        if (typeof process === "undefined")
            return null;
        cwd = process.cwd();
    }
    const configFilePath = findConfigFilePath(cwd);
    if (!configFilePath)
        return null;
    if (caches.has(configFilePath)) {
        return caches.get(configFilePath) || null;
    }
    const code = fs.readFileSync(configFilePath, "utf8");
    const config = parseConfig(code);
    caches.set(configFilePath, config);
    return config;
}
/**
 * Searches from the current working directory up until finding the config filename.
 * @param {string} cwd The current working directory to search from.
 * @returns {string|undefined} The file if found or `undefined` if not.
 */
function findConfigFilePath(cwd) {
    let directory = path.resolve(cwd);
    const { root } = path.parse(directory);
    const stopAt = path.resolve(directory, root);
    while (directory !== stopAt) {
        const target = path.resolve(directory, "svelte.config.js");
        const stat = fs.existsSync(target)
            ? fs.statSync(target, {
                throwIfNoEntry: false,
            })
            : null;
        if (stat?.isFile()) {
            return target;
        }
        const next = path.dirname(directory);
        if (next === directory)
            break;
        directory = next;
    }
    return null;
}
