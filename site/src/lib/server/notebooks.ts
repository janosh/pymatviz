import { readdirSync, readFileSync } from 'node:fs'

const examples_dir = new URL(`../examples/`, `file://${process.cwd()}/`)

const example_files = (extension: string) =>
  readdirSync(examples_dir)
    .filter((file_name) => file_name.endsWith(extension))
    .sort()

export const notebook_routes = (extension = `.ipynb`) =>
  example_files(extension).map(
    (file_name) => `/notebooks/${file_name.slice(0, -extension.length)}`,
  )

export const notebook_entries = () =>
  example_files(`.html`).map((file_name) => ({
    slug: file_name.slice(0, -`.html`.length),
  }))

export const read_notebook_html = (slug: string) => {
  try {
    return readFileSync(new URL(`${slug}.html`, examples_dir), `utf8`)
  } catch {
    return null
  }
}
