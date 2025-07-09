import SafeParser from 'postcss-safe-parser/lib/safe-parser.js';
declare class TemplateSafeParser extends SafeParser {
    protected createTokenizer(): void;
}
export default TemplateSafeParser;
