import type { Context } from "./context/index.js";
/**
 * Svelte parse errors.
 */
export declare class ParseError extends SyntaxError {
    index: number;
    lineNumber: number;
    column: number;
    /**
     * Initialize this ParseError instance.
     */
    constructor(message: string, offset: number, ctx: Context);
}
