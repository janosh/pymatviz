import { beginShared, terminateShared } from '../shared/index.js';
export * as meta from '../meta.js';
/** preprocess */
export function preprocess(code, filename) {
    if (filename) {
        beginShared(filename);
    }
    return [code];
}
/** postprocess */
export function postprocess([messages], filename) {
    const shared = terminateShared(filename);
    if (shared) {
        return filter(messages, shared);
    }
    return messages;
}
export const supportsAutofix = true;
/** Filter  */
function filter(messages, shared) {
    if (shared.commentDirectives.length === 0) {
        return messages;
    }
    let filteredMessages = messages;
    for (const cd of shared.commentDirectives) {
        filteredMessages = cd.filterMessages(filteredMessages);
    }
    return filteredMessages;
}
