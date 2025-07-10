import './rule-types.js';
import { rules as ruleList } from './utils/rules.js';
import base from './configs/flat/base.js';
import recommended from './configs/flat/recommended.js';
import prettier from './configs/flat/prettier.js';
import all from './configs/flat/all.js';
import * as processor from './processor/index.js';
import * as metaModule from './meta.js';
export const configs = {
    base,
    recommended,
    prettier,
    all,
    // For backward compatibility
    'flat/base': base,
    'flat/recommended': recommended,
    'flat/prettier': prettier,
    'flat/all': all
};
export const rules = ruleList.reduce((obj, r) => {
    obj[r.meta.docs.ruleName] = r;
    return obj;
}, {});
export const meta = { ...metaModule };
export const processors = {
    '.svelte': processor,
    svelte: processor
};
