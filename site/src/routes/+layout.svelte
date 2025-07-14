<script lang="ts">
  import { goto } from '$app/navigation'
  import { page } from '$app/state'
  import { repository } from '$site/package.json'
  import Icon from '@iconify/svelte'
  import type { Snippet } from 'svelte'
  import { CmdPalette, GitHubCorner } from 'svelte-multiselect'
  import Toc from 'svelte-toc'
  import '../app.css'

  interface Props {
    children?: Snippet
  }
  let { children }: Props = $props()

  let headingSelector = $derived(
    `main :is(${page.url.pathname === `/api` ? `h1, h2, h3, h4` : `h2`}):not(.toc-exclude)`,
  )

  const file_routes = Object.keys(import.meta.glob(`./**/+page.{svx,svelte,md}`))
    .filter((key) => !key.includes(`/[`))
    .map((filename) => {
      const parts = filename.split(`/`)
      return `/` + parts.slice(1, -1).join(`/`)
    })

  const notebooks = Object.keys(
    import.meta.glob(`$root/examples/*.html`, {
      eager: true,
      query: `?url`,
      import: `default`,
    }),
  ).map((path) => {
    const filename = path.split(`/`).at(-1)?.replace(`.html`, ``)
    return `/notebooks/${filename}`
  })
  const actions = file_routes.concat(notebooks).map((name) => {
    return { label: name, action: () => goto(name.toLowerCase()) }
  })
</script>

<CmdPalette {actions} placeholder="Go to..." />

<Toc {headingSelector} breakpoint={1250} warnOnEmpty={false} />

{#if page.url.pathname !== `/`}
  <a href="/" aria-label="Back to index page">&laquo; home</a>
{/if}

<GitHubCorner href={repository} />

{@render children?.()}

<footer>
  <nav>
    <a href="{repository}/issues">
      <Icon icon="octicon:mark-github" inline />&ensp;Issues
    </a>
    <a href="{repository}/discussion"><Icon icon="mdi:chat" inline />&ensp;Discussion</a>
  </nav>
  <img
    src="https://github.com/janosh/pymatviz/raw/main/site/static/favicon.svg"
    alt="Logo"
    height="40px"
  />
  <strong>pymatviz</strong>
</footer>

<style>
  a[href='/'] {
    font-size: 15pt;
    position: absolute;
    top: 2em;
    left: 2em;
    background-color: rgba(255, 255, 255, 0.1);
    padding: 1pt 5pt;
    border-radius: 3pt;
    transition: 0.2s;
  }
  a[href='/']:hover {
    background-color: rgba(255, 255, 255, 0.2);
  }
  footer {
    padding: 3vh 3vw;
    background: #00061a;
    text-align: center;
  }
  footer nav {
    margin: 2em;
    display: flex;
    gap: 2em;
    place-content: center;
    flex-wrap: wrap;
  }
  strong {
    font-size: 20px;
    vertical-align: 16px;
  }
</style>
