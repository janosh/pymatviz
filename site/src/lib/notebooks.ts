// Shared notebook path extraction from glob
const notebook_html_files: Record<string, string> = import.meta.glob(
  `$root/examples/*.html`,
  { eager: true, query: `?url`, import: `default` },
)

export const notebook_paths: string[] = Object.keys(notebook_html_files)

export const notebook_routes: string[] = notebook_paths.map((path) => {
  const filename = path.split(`/`).at(-1)?.replace(`.html`, ``)
  return `/notebooks/${filename}`
})
