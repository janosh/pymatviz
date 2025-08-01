import * as AST from "./ast/index.js";
import { traverseNodes } from "./traverse.js";
import { ParseError } from "./errors.js";
export { parseForESLint, type StyleContext, type StyleContextNoStyleElement, type StyleContextParseError, type StyleContextSuccess, type StyleContextUnknownLang, } from "./parser/index.js";
export { name } from "./meta.js";
export declare const meta: {
    name: "svelte-eslint-parser";
    version: "1.2.0";
};
export type { SvelteConfig } from "./svelte-config/index.js";
export { AST, ParseError };
export declare const VisitorKeys: import("eslint").SourceCode.VisitorKeys;
export { traverseNodes };
