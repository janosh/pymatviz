import { readdirSync, readFileSync } from 'node:fs'

const examples_dir = new URL(`../examples/`, `file://${process.cwd()}/`)

const slugs = (extension: string) =>
  readdirSync(examples_dir)
    .filter((file_name) => file_name.endsWith(extension))
    .toSorted()
    .map((file_name) => file_name.slice(0, -extension.length))

const href_label = (slug: string): [string, string] => [
  `/notebooks/${slug}`,
  slug.replaceAll(`_`, ` `),
]

export const notebook_routes = (extension = `.ipynb`) =>
  slugs(extension).map((slug) => `/notebooks/${slug}`)

export const notebook_subpages = (): [string, string, string][] =>
  slugs(`.ipynb`).map((slug) => {
    const [href, title] = href_label(slug)
    return [title, href, `${slug}.ipynb`]
  })

export const notebook_prev_next = (): [string, string][] => slugs(`.html`).map(href_label)

export const notebook_entries = () => slugs(`.html`).map((slug) => ({ slug }))

export const read_notebook_html = (slug: string) => {
  try {
    return readFileSync(new URL(`${slug}.html`, examples_dir), `utf8`)
  } catch {
    return null
  }
}
