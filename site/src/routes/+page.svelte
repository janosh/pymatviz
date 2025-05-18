<script lang="ts">
  import Readme from '$root/readme.md'
</script>

<main>
  <Readme>
    <span slot="how-to-cite">
      If you use <code>pymatviz</code> in your research,
      <a href="#how-to-cite-pymatviz">see how to cite</a>.
    </span>
  </Readme>
</main>

<style>
  :global(h1[align='center']) {
    display: flex;
    font-size: clamp(2rem, 2rem + 2vw, 3rem);
    place-items: center;
    place-content: center;
    gap: 6pt;
  }
  :global(table) {
    max-width: var(--max-main-width);
    table-layout: fixed;
  }
</style>
