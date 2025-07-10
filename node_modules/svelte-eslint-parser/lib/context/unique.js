export class UniqueIdGenerator {
    constructor() {
        this.uniqueIdSeq = 1;
        this.usedUniqueIds = new Set();
    }
    generate(base, ...texts) {
        const hasId = (id) => this.usedUniqueIds.has(id) || texts.some((t) => t.includes(id));
        let candidate = `$_${base.replace(/\W/g, "_")}${this.uniqueIdSeq++}`;
        while (hasId(candidate)) {
            candidate = `$_${base.replace(/\W/g, "_")}${this.uniqueIdSeq++}`;
        }
        this.usedUniqueIds.add(candidate);
        return candidate;
    }
}
