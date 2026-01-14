import type { ESLintExtendedProgram } from "../index.js";
import type { NormalizedParserOptions } from "../parser-options.js";
import type { SvelteParseContext } from "../svelte-parse-context.js";
import type { AnalyzeTypeScriptContext } from "./analyze/index.js";
/**
 * Parse for TypeScript in <script>
 */
export declare function parseTypeScriptInSvelte(code: {
    script: string;
    render: string;
    rootScope: string;
}, attrs: Record<string, string | undefined>, parserOptions: NormalizedParserOptions, context: AnalyzeTypeScriptContext): ESLintExtendedProgram;
/**
 * Parse for TypeScript
 */
export declare function parseTypeScript(code: string, attrs: Record<string, string | undefined>, parserOptions: NormalizedParserOptions, svelteParseContext: SvelteParseContext): ESLintExtendedProgram;
