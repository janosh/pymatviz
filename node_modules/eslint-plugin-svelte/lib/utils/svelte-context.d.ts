import type { RuleContext } from '../types.js';
export type SvelteContext = ({
    svelteVersion: '3/4';
    svelteFileType: '.svelte' | null;
    runes: null;
} | ({
    svelteVersion: '5';
} & ({
    svelteFileType: '.svelte' | '.svelte.[js|ts]';
    /** If a user uses a parser other than `svelte-eslint-parser`, `undetermined` will be set. */
    runes: boolean | 'undetermined';
} | {
    /** e.g. `foo.js` / `package.json` */
    svelteFileType: null;
    runes: null;
})) | {
    /** For projects that do not use Svelte. */
    svelteVersion: null;
    svelteFileType: null;
    runes: null;
}) & {
    svelteKitVersion: '1.0.0-next' | '1' | '2' | null;
    svelteKitFileType: '+page.svelte' | '+page.[js|ts]' | '+page.server.[js|ts]' | '+error.svelte' | '+layout.svelte' | '+layout.[js|ts]' | '+layout.server.[js|ts]' | '+server.[js|ts]' | null;
};
export declare function getSvelteVersion(): SvelteContext['svelteVersion'];
export declare function getSvelteContext(context: RuleContext): SvelteContext | null;
