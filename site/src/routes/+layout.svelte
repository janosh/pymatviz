<script lang="ts">
  import { afterNavigate, goto } from '$app/navigation'
  import { asset } from '$app/paths'
  import { page } from '$app/state'
  import { repository } from '$site/package.json'
  import type { Snippet } from 'svelte'
  import {
    CopyButton,
    Footer,
    GitHubCorner,
    Nav,
    PageSearch,
    Toc,
    type FooterLink,
  } from 'svelte-widgets'
  import { highlight_matches } from 'svelte-widgets/attachments'
  import { heading_anchors } from 'svelte-widgets/heading-anchors'
  // oxlint-disable-next-line import/no-unassigned-import -- global app styles
  import '../app.css'

  let { children, data }: { children?: Snippet; data: { notebook_routes: string[] } } =
    $props()
  let page_search_query = $state(``)

  const heading_selector = $derived(
    `main :is(${
      page.url.pathname === `/api` ? `h1, h2, h3, h4` : `h2`
    }):not(.toc-exclude, :where(.subpage-grid *))`,
  )

  const nav_labels: Record<string, string> = {
    '/': `Home`,
    '/api': `API`,
    '/notebooks': `Notebooks`,
    '/plots': `Plots`,
    '/changelog': `Changelog`,
  }
  const page_title = $derived(nav_labels[page.url.pathname] ?? `pymatviz`)

  const page_routes = Object.keys(import.meta.glob(`./**/+page.{svx,svelte,md}`))
    .filter((route) => !route.includes(`/[`))
    .map((route) => route.replace(/^\./u, ``).replace(/\/\+page\.\w+$/u, ``) || `/`)
  const fallback_actions = $derived(
    [...new Set([...page_routes, ...data.notebook_routes])].map((name) => ({
      label: name,
      action: () => goto(name),
    })),
  )

  const footer_links: FooterLink[] = [
    { href: `${repository}/issues`, label: `Issues`, icon: `Issues` },
    { href: `${repository}/discussions`, label: `Discussion`, icon: `Comment` },
  ]

  afterNavigate(() => (page_search_query = ``))
</script>

<svelte:head>
  <title>{page.url.pathname === `/` ? `pymatviz` : `${page_title} · pymatviz`}</title>
  <meta data-pagefind-meta="title[content]" content={page_title} />
</svelte:head>

<PageSearch
  {fallback_actions}
  navigate={async (url, { query }) => {
    await goto(url)
    page_search_query = ``
    queueMicrotask(() => (page_search_query = query))
  }}
  strip_html_suffix
  pagefind_path={asset(`/pagefind/pagefind.js`)}
/>

<CopyButton global global_selector="pre:not(li > pre) > code" />

<Nav class="site-nav" routes={Object.keys(nav_labels)} labels={nav_labels} {page} />

<Toc headingSelector={heading_selector} breakpoint={1500} />

<GitHubCorner href={repository} />

<main
  data-pagefind-body
  {@attach heading_anchors()}
  {@attach highlight_matches({ query: page_search_query, duration_ms: 8000 })}
>
  {@render children?.()}
</main>

<Footer links={footer_links}>
  <img src="/favicon.svg" alt="Logo" height="40px" />
  <strong>pymatviz</strong>
</Footer>
