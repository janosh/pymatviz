import { CommentDirectives } from './comment-directives.js';
export class Shared {
    constructor() {
        this.commentDirectives = [];
    }
    newCommentDirectives(options) {
        const directives = new CommentDirectives(options);
        this.commentDirectives.push(directives);
        return directives;
    }
}
const sharedMap = new Map();
/** Start sharing and make the data available. */
export function beginShared(filename) {
    sharedMap.set(filename, new Shared());
}
/** Get the shared data and end the sharing. */
export function terminateShared(filename) {
    const result = sharedMap.get(filename);
    sharedMap.delete(filename);
    return result ?? null;
}
/** If sharing has started, get the shared data. */
export function getShared(filename) {
    return sharedMap.get(filename) ?? null;
}
