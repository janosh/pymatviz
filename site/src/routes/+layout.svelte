<script lang="ts">
  import { goto } from '$app/navigation'
  import { page } from '$app/state'
  import { repository } from '$site/package.json'
  import Icon from '@iconify/svelte'
  import type { Snippet } from 'svelte'
  import { CmdPalette, GitHubCorner } from 'svelte-multiselect'
  import { heading_anchors } from 'svelte-multiselect/heading-anchors'
  import Toc from 'svelte-toc'
  // oxlint-disable-next-line import/no-unassigned-import -- global app styles
  import '../app.css'

  let {
    children,
    data,
  }: {
    children?: Snippet
    data: { notebook_routes: string[] }
  } = $props()
  let toc_desktop = $state(true)

  let headingSelector = $derived(
    `main :is(${
      page.url.pathname === `/api` ? `h1, h2, h3, h4` : `h2`
    }):not(.toc-exclude)`,
  )

  const file_routes = Object.keys(import.meta.glob(`./**/+page.{svx,svelte,md}`))
    .filter((key) => !key.includes(`/[`))
    .map((filename) => filename.replace(/^\./u, ``).replace(/\/\+page\.\w+$/u, ``) || `/`)

  let actions = $derived(
    [...new Set([...file_routes, ...data.notebook_routes])].map((name) => ({
      label: name,
      action: () => goto(name),
    })),
  )
</script>

<CmdPalette {actions} placeholder="Go to..." />

<Toc
  {headingSelector}
  breakpoint={1500}
  bind:desktop={toc_desktop}
  asideProps={{
    style: toc_desktop
      ? `position: fixed; font-size: 0.6em; top: 5em; left: calc(50vw + var(--max-main-width) / 2 + 15em)`
      : ``,
  }}
  navProps={{ style: toc_desktop ? `` : `padding-left: 9pt` }}
  --toc-mobile-bg="#0d1a1d"
  --toc-mobile-shadow="0 0 1em 0 black"
  --toc-active-border="solid var(--blue)"
  --toc-active-border-width="0 0 0 2pt"
/>

{#if page.url.pathname !== `/`}
  <a href="/" aria-label="Back to index page">&laquo; home</a>
{/if}

<GitHubCorner href={repository} />

<main {@attach heading_anchors()}>
  {@render children?.()}
</main>

<footer>
  <nav>
    <a href="{repository}/issues">
      <Icon icon="octicon:mark-github" inline />&ensp;Issues
    </a>
    <a href="{repository}/discussion"><Icon icon="mdi:chat" inline />&ensp;Discussion</a>
  </nav>
  <img src="/favicon.svg" alt="Logo" height="40px" />
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
