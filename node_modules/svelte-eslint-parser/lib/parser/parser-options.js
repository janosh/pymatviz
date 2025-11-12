import fs from "fs";
import path from "path";
import { isTSESLintParserObject, maybeTSESLintParserObject, } from "./parser-object.js";
import { getParserForLang } from "./resolve-parser.js";
/** Normalize parserOptions */
export function normalizeParserOptions(options) {
    const parserOptions = {
        ecmaVersion: 2020,
        sourceType: "module",
        loc: true,
        range: true,
        raw: true,
        tokens: true,
        comment: true,
        eslintVisitorKeys: true,
        eslintScopeManager: true,
        ...(options || {}),
    };
    parserOptions.sourceType = "module";
    if (parserOptions.ecmaVersion <= 5 || parserOptions.ecmaVersion == null) {
        parserOptions.ecmaVersion = 2015;
    }
    return parserOptions;
}
const TS_PARSER_NAMES = [
    "@typescript-eslint/parser",
    "typescript-eslint-parser-for-extra-files",
];
export function isTypeScript(parserOptions, lang) {
    if (!lang) {
        return false;
    }
    const parserValue = getParserForLang(lang, parserOptions?.parser);
    if (typeof parserValue !== "string") {
        return (maybeTSESLintParserObject(parserValue) ||
            isTSESLintParserObject(parserValue));
    }
    const parserName = parserValue;
    if (TS_PARSER_NAMES.includes(parserName)) {
        return true;
    }
    if (TS_PARSER_NAMES.some((nm) => parserName.includes(nm))) {
        let targetPath = parserName;
        while (targetPath) {
            const pkgPath = path.join(targetPath, "package.json");
            if (fs.existsSync(pkgPath)) {
                try {
                    return TS_PARSER_NAMES.includes(JSON.parse(fs.readFileSync(pkgPath, "utf-8"))?.name);
                }
                catch {
                    return false;
                }
            }
            const parent = path.dirname(targetPath);
            if (targetPath === parent) {
                break;
            }
            targetPath = parent;
        }
    }
    return false;
}
/**
 * Remove typing-related options from parser options.
 *
 * Allows the typescript-eslint parser to parse a file without
 * trying to collect typing information from TypeScript.
 *
 * See https://typescript-eslint.io/packages/parser#withoutprojectparseroptionsparseroptions
 */
export function withoutProjectParserOptions(options) {
    const { project: _strippedProject, projectService: _strippedProjectService, EXPERIMENTAL_useProjectService: _strippedExperimentalUseProjectService, ...result } = options;
    return result;
}
