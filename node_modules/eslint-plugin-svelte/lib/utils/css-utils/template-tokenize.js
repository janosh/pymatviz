import tokenize from 'postcss/lib/tokenize';
/** Tokenize */
function templateTokenize(...args) {
    const tokenizer = tokenize(...args);
    /** nextToken */
    function nextToken(...args) {
        const returned = [];
        let token, lastPos;
        let depth = 0;
        while ((token = tokenizer.nextToken(...args))) {
            if (token[0] !== 'word') {
                if (token[0] === '{') {
                    ++depth;
                }
                else if (token[0] === '}') {
                    --depth;
                }
            }
            if (depth || returned.length) {
                lastPos = token[3] || token[2] || lastPos;
                returned.push(token);
            }
            if (!depth) {
                break;
            }
        }
        if (returned.length) {
            token = ['word', returned.map((token) => token[1]).join(''), returned[0][2], lastPos];
        }
        return token;
    }
    return Object.assign({}, tokenizer, {
        nextToken
    });
}
export default templateTokenize;
