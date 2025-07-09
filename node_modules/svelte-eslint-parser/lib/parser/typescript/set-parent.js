import { traverseNodes } from "../../index.js";
export function setParent(result) {
    if (result.ast.body.some((node) => node.parent)) {
        return;
    }
    traverseNodes(result.ast, {
        visitorKeys: result.visitorKeys,
        enterNode(node, parent) {
            node.parent = parent;
        },
        leaveNode() {
            // noop
        },
    });
}
