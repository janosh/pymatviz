<script lang="ts">
  export let title: string
  export let file: string
  export let code: string
  export let highlighted: string
</script>

<h2>{title}</h2>

<section>
  {@html highlighted}
  <img src="/assets/{file}" alt={title} />
</section>

<style>
  h2 {
    text-align: center;
    margin: 1em auto 3pt;
  }
  section {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-gap: 2em;
    width: 100%;
    max-width: 90vw;
    margin: 1em auto;
  }
  section > :global(pre) {
    width: 100%;
    font-size: 9pt;
    height: fit-content;
    margin: 0;
  }
  section > img {
    width: 100%;
    max-height: 50vh;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 4pt;
  }
</style>
