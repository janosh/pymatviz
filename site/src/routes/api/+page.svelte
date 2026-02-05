<script lang="ts">
  export let data
</script>

<h1 class="toc-exclude">API</h1>

{@html data.html}

<style>
  /* select all but first module h1s */
  :global(h1[id^='module-']:not(:nth-of-type(2))) {
    margin: 2em 0 0;
  }
  :global(hr) {
    border: none;
  }
  :global(code) {
    line-height: 1em;
    border-radius: 4pt;
  }
</style>
