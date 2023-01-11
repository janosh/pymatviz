import adapter from '@sveltejs/adapter-static'
import { s } from 'hastscript'
import { mdsvex } from 'mdsvex'
import link_headings from 'rehype-autolink-headings'
import heading_slugs from 'rehype-slug'
import preprocess from 'svelte-preprocess'

const rehypePlugins = [
  heading_slugs,
  [
    link_headings,
    {
      behavior: `append`,
      test: [`h2`, `h3`, `h4`, `h5`, `h6`], // don't auto-link <h1>
      content: s(
        `svg`,
        { width: 16, height: 16, viewBox: `0 0 16 16` },
        // symbol #octicon-link defined in app.html
        s(`use`, { 'xlink:href': `#octicon-link` })
      ),
    },
  ],
]

/** @type {import('@sveltejs/kit').Config} */
export default {
  extensions: [`.svelte`, `.svx`, `.md`],

  preprocess: [
    preprocess(),
    mdsvex({
      rehypePlugins,
      extensions: [`.svx`, `.md`],
    }),
  ],

  kit: {
    adapter: adapter(),

    prerender: {
      handleHttpError: `warn`,
    },
  },
}
