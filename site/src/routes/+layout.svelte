<script lang="ts">
  import { goto } from '$app/navigation'
  import { page } from '$app/state'
  import { notebook_routes } from '$lib/notebooks'
  import { repository } from '$site/package.json'
  import Icon from '@iconify/svelte'
  import type { Snippet } from 'svelte'
  import { CmdPalette, GitHubCorner } from 'svelte-multiselect'
  import { heading_anchors } from 'svelte-multiselect/heading-anchors'
  import Toc from 'svelte-toc'
  import '../app.css'

  let { children }: { children?: Snippet } = $props()

  let headingSelector = $derived(
    `main :is(${
      page.url.pathname === `/api` ? `h1, h2, h3, h4` : `h2`
    }):not(.toc-exclude)`,
  )

  const file_routes = Object.keys(import.meta.glob(`./**/+page.{svx,svelte,md}`))
    .filter((key) => !key.includes(`/[`))
    .map((filename) =>
      filename.replace(/^\./, ``).replace(/\/\+page\.\w+$/, ``) || `/`
    )

  const actions = file_routes.concat(notebook_routes).map((name) => ({
    label: name,
    action: () => goto(name),
  }))
</script>

<CmdPalette {actions} placeholder="Go to..." />

<Toc
  {headingSelector}
  breakpoint={1250}
  warnOnEmpty={false}
  --toc-mobile-bg="#0d1a1d"
  --toc-mobile-shadow="0 0 1em 0 black"
  --toc-title-padding="0 0 0 3pt"
  --toc-li-padding="2pt 1ex"
  --toc-mobile-btn-color="white"
  --toc-mobile-btn-bg="teal"
  --toc-mobile-btn-padding="1pt 2pt"
  --toc-desktop-nav-margin="0 0 0 1em"
  --toc-min-width="15em"
  --toc-active-border="solid cornflowerblue"
  --toc-active-border-width="0 0 0 2pt"
  --toc-active-bg="none"
  --toc-active-border-radius="0"
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
  :global(aside.toc.desktop) {
    position: fixed;
    top: 3em;
    right: 6em;
    max-width: 300px;
    font-size: 0.6em;
  }
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
