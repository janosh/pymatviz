import { type UserOptionParser } from "./resolve-parser.js";
export type NormalizedParserOptions = {
    parser?: UserOptionParser;
    project?: string | string[] | null;
    projectService?: unknown;
    EXPERIMENTAL_useProjectService?: unknown;
    ecmaVersion: number | "latest";
    sourceType: "module" | "script";
    ecmaFeatures?: {
        globalReturn?: boolean | undefined;
        impliedStrict?: boolean | undefined;
        jsx?: boolean | undefined;
        experimentalObjectRestSpread?: boolean | undefined;
        [key: string]: any;
    };
    svelteFeatures?: {
        runes?: boolean;
    };
    loc: boolean;
    range: boolean;
    raw: boolean;
    tokens: boolean;
    comment: boolean;
    eslintVisitorKeys: boolean;
    eslintScopeManager: boolean;
    filePath?: string;
};
/** Normalize parserOptions */
export declare function normalizeParserOptions(options: any): NormalizedParserOptions;
export declare function isTypeScript(parserOptions: NormalizedParserOptions, lang: string | undefined): boolean;
/**
 * Remove typing-related options from parser options.
 *
 * Allows the typescript-eslint parser to parse a file without
 * trying to collect typing information from TypeScript.
 *
 * See https://typescript-eslint.io/packages/parser#withoutprojectparseroptionsparseroptions
 */
export declare function withoutProjectParserOptions(options: NormalizedParserOptions): NormalizedParserOptions;
