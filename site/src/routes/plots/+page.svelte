<script lang="ts">
  import { Masonry } from 'svelte-widgets'

  type PlotFig = { id: string; src: string; title: string }

  const figs: PlotFig[] = Object.entries(
    import.meta.glob<string>(`$root/assets/svg/*.svg`, {
      eager: true,
      query: `?url`,
      import: `default`,
    }),
  ).map(([path, src]) => ({
    id: path,
    src,
    title: /(?<stem>[^/]+)\.svg$/u.exec(path)?.groups?.stem?.replaceAll(`-`, ` `) ?? path,
  }))
</script>

<h1>Figures</h1>

<Masonry items={figs} minColWidth={300} gap={32}>
  {#snippet children({ item, idx }: { item: PlotFig; idx: number })}
    <article>
      <span>{idx + 1}</span>
      <h3>{item.title}</h3>
      <img src={item.src} alt={item.title} />
    </article>
  {/snippet}
</Masonry>

<style>
  h1 {
    margin: 0 0 1em;
  }
  article {
    position: relative;
    background-color: rgba(255, 255, 255, 0.05);
    padding: 0 1em 1em;
    border-radius: 4pt;
  }
  h3 {
    text-align: center;
    margin: 1em;
    text-transform: capitalize;
  }
  span {
    position: absolute;
    font-weight: lighter;
    margin: 0.5em 0;
  }
  img {
    width: 100%;
  }
  :global(main:has(.masonry)) {
    max-width: min(90vw, 1200px);
  }
</style>
