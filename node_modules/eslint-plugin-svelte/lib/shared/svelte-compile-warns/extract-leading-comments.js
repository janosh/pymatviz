import { isOpeningParenToken } from '@eslint-community/eslint-utils';
/** Extract comments */
export function extractLeadingComments(context, node) {
    const sourceCode = context.sourceCode;
    const beforeToken = sourceCode.getTokenBefore(node, {
        includeComments: false,
        filter(token) {
            if (isOpeningParenToken(token)) {
                return false;
            }
            const astToken = token;
            if (astToken.type === 'HTMLText') {
                return false;
            }
            return astToken.type !== 'HTMLComment';
        }
    });
    if (beforeToken) {
        return sourceCode
            .getTokensBetween(beforeToken, node, { includeComments: true })
            .filter(isComment);
    }
    return sourceCode.getTokensBefore(node, { includeComments: true }).filter(isComment);
}
/** Checks whether given token is comment token */
function isComment(token) {
    return token.type === 'HTMLComment' || token.type === 'Block' || token.type === 'Line';
}
