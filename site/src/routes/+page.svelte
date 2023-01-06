<script lang="ts">
  import Readme from '$root/readme.md'
</script>

<Readme />

<style>
  :global(h1[align='center']) {
    display: flex;
    font-size: clamp(2rem, 2rem + 2vw, 3rem);
    place-items: center;
    place-content: center;
    gap: 6pt;
  }
</style>
