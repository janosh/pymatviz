<script lang="ts">
  import { page } from '$app/state'
  import { PrevNext, type LinkItem } from 'svelte-widgets'

  let { data }: { data: { html: string; items: LinkItem[] } } = $props()
</script>

{@html data.html}

<PrevNext items={data.items} current={page.url.pathname} />
