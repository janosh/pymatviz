<script lang="ts">
  import { repository } from '$site/package.json'
  import Icon from '@iconify/svelte'
</script>

<footer>
  Maintained by Janosh Riebesell
  <a href="mailto:janosh@lbl.gov"><Icon icon="mdi:email" inline /></a>
  on
  <a href={repository}><Icon icon="octicon:mark-github" inline /></a>
</footer>

<style>
  footer {
    display: flex;
    gap: 1ex;
    place-content: center;
    place-items: center;
    padding: 3vh 3vw;
    background: #00061a;
  }
</style>
