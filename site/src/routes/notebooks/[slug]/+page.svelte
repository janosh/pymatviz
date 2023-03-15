<script lang="ts">
  import { page } from '$app/stores'
  import { PrevNext } from 'svelte-zoo'

  export let data
</script>

<main>
  {@html data.html}

  <PrevNext items={data.routes} current={$page.url.pathname} />
</main>
