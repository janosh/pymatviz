/**
 * Svelte parse errors.
 */
export class ParseError extends SyntaxError {
    /**
     * Initialize this ParseError instance.
     */
    constructor(message, offset, ctx) {
        super(message);
        this.index = offset;
        const loc = ctx.getLocFromIndex(offset);
        this.lineNumber = loc.line;
        this.column = loc.column;
    }
}
