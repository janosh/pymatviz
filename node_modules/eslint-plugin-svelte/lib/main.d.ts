import './rule-types.js';
import * as processor from './processor/index.js';
import type { Rule } from 'eslint';
export declare const configs: {
    base: import("eslint").Linter.Config<import("eslint").Linter.RulesRecord>[];
    recommended: import("eslint").Linter.Config<import("eslint").Linter.RulesRecord>[];
    prettier: import("eslint").Linter.Config<import("eslint").Linter.RulesRecord>[];
    all: import("eslint").Linter.Config<import("eslint").Linter.RulesRecord>[];
    'flat/base': import("eslint").Linter.Config<import("eslint").Linter.RulesRecord>[];
    'flat/recommended': import("eslint").Linter.Config<import("eslint").Linter.RulesRecord>[];
    'flat/prettier': import("eslint").Linter.Config<import("eslint").Linter.RulesRecord>[];
    'flat/all': import("eslint").Linter.Config<import("eslint").Linter.RulesRecord>[];
};
export declare const rules: Record<string, Rule.RuleModule>;
export declare const meta: {
    name: "eslint-plugin-svelte";
    version: "3.10.1";
};
export declare const processors: {
    '.svelte': typeof processor;
    svelte: typeof processor;
};
