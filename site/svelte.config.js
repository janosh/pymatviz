import adapter from '@sveltejs/adapter-static'
import { mdsvex } from 'mdsvex'
import { heading_ids } from 'svelte-multiselect/heading-anchors'
import { sveltePreprocess } from 'svelte-preprocess'

const { default: pkg } = await import(`./package.json`, {
  with: { type: `json` },
})

/** @type {import('@sveltejs/kit').Config} */
export default {
  extensions: [`.svelte`, `.svx`, `.md`],

  preprocess: [
    // replace readme links to docs with site-internal links
    // (which don't require browser navigation)
    sveltePreprocess({ replace: [[pkg.homepage, ``]] }),
    mdsvex({ extensions: [`.svx`, `.md`] }),
    heading_ids(),
  ],

  kit: {
    adapter: adapter(),

    alias: {
      $src: `./src`,
      $site: `.`,
      $root: `..`,
    },
  },
}
