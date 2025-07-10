import SafeParser from 'postcss-safe-parser/lib/safe-parser.js';
import templateTokenize from './template-tokenize.js';
class TemplateSafeParser extends SafeParser {
    createTokenizer() {
        this.tokenizer = templateTokenize(this.input, { ignoreErrors: true });
    }
}
export default TemplateSafeParser;
