import { VERSION as compilerVersion } from "svelte/compiler";
export { compilerVersion };
const verStrings = compilerVersion.split(".");
export const svelteVersion = {
    gte(v) {
        return Number(verStrings[0]) >= v;
    },
};
