<script lang="ts">
  import Changelog from '$root/changelog.md'
  import Toc from 'svelte-toc'
</script>

<Changelog />

<Toc headingSelector="main > :where(h4)" breakpoint={1250} />
