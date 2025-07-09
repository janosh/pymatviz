import type { SvelteHTMLNode } from "./html.js";
import type { SvelteScriptNode } from "./script.js";
export * from "./common.js";
export * from "./html.js";
export * from "./script.js";
export type SvelteNode = SvelteHTMLNode | SvelteScriptNode;
