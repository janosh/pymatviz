<script lang="ts">
  import { page } from '$app/state'
  import { PrevNext } from 'svelte-multiselect'

  let { data }: { data: { html: string; slug: string; path: string; routes: string[] } } = $props()
</script>

{@html data.html}

<PrevNext items={data.routes} current={page.url.pathname} />
