import { getSvelteContext } from '../utils/svelte-context.js';
function doesNotSatisfy(actual, expected) {
    if (expected == null || expected.length === 0) {
        return false;
    }
    return !expected.includes(actual);
}
function satisfiesCondition(condition, svelteContext) {
    if (doesNotSatisfy(svelteContext.svelteVersion, condition.svelteVersions) ||
        doesNotSatisfy(svelteContext.svelteFileType, condition.svelteFileTypes) ||
        doesNotSatisfy(svelteContext.runes, condition.runes) ||
        doesNotSatisfy(svelteContext.svelteKitVersion, condition.svelteKitVersions) ||
        doesNotSatisfy(svelteContext.svelteKitFileType, condition.svelteKitFileTypes)) {
        return false;
    }
    return true;
}
// export for testing
export function shouldRun(svelteContext, conditions) {
    // If svelteContext is null, it means the rule might be executed based on the analysis result of a different parser.
    // In this case, always execute the rule.
    if (svelteContext == null || conditions == null || conditions.length === 0) {
        return true;
    }
    for (const condition of conditions) {
        if (satisfiesCondition(condition, svelteContext)) {
            return true;
        }
    }
    return false;
}
/**
 * Define the rule.
 * @param ruleName ruleName
 * @param rule rule module
 */
export function createRule(ruleName, rule) {
    return {
        meta: {
            ...rule.meta,
            docs: {
                ...rule.meta.docs,
                url: `https://sveltejs.github.io/eslint-plugin-svelte/rules/${ruleName}/`,
                ruleId: `svelte/${ruleName}`,
                ruleName
            }
        },
        create(context) {
            const { conditions } = rule.meta;
            const svelteContext = getSvelteContext(context);
            if (!shouldRun(svelteContext, conditions)) {
                return {};
            }
            return rule.create(context);
        }
    };
}
