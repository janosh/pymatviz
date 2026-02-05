<script lang="ts">
  import { page } from '$app/state'
  import { PrevNext } from 'svelte-multiselect'

  export let data
</script>

{@html data.html}

<PrevNext items={data.routes} current={page.url.pathname} />
