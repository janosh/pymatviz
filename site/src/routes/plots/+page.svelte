<script lang="ts">
  const figs = import.meta.glob(`$root/assets/*.svg`, {
    eager: true,
    as: `url`,
  })
</script>

<h1>Figures</h1>

<ul>
  {#each Object.entries(figs) as [alt, src], idx}
    {@const filename = alt.split(`/`).at(-1)?.split(`.`)[0]}
    <li>
      <span>{idx + 1}</span>
      <h3>{filename?.replaceAll(`-`, ` `)}</h3>
      {#if src.endsWith(`.svelte`)}
        {#await import(src) then component}
          <svelte:component this={component.default} />
        {/await}
      {:else if src.endsWith(`.pdf`)}
        <embed {src} width="100%" height="500px" />
      {:else}
        <img {src} {alt} />
      {/if}
    </li>
  {/each}
</ul>

<style>
  h1 {
    margin: 3em 0 1em;
  }
  img {
    width: 100%;
  }
  ul {
    list-style: none;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 2em;
    max-width: min(90vw, 1200px);
    margin: 0 auto;
  }
  ul > li {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 0 1em;
    border-radius: 4pt;
  }
  ul > li > h3 {
    text-align: center;
    margin: 1em;
    text-transform: capitalize;
  }
  ul > li > span {
    position: absolute;
    font-weight: lighter;
    margin: 0.5em 0;
  }
</style>
