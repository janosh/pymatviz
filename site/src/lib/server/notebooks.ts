import type { LinkItem, Subpage } from 'svelte-widgets'
import { readdirSync, readFileSync } from 'node:fs'

const examples_dir = new URL(`../examples/`, `file://${process.cwd()}/`)

const slugs = (extension: string) =>
  readdirSync(examples_dir)
    .filter((file_name) => file_name.endsWith(extension))
    .toSorted()
    .map((file_name) => file_name.slice(0, -extension.length))

const notebook_link = (slug: string): LinkItem => ({
  href: `/notebooks/${slug}`,
  label: slug.replaceAll(`_`, ` `),
})

export const notebook_subpages = (): Subpage[] =>
  slugs(`.ipynb`).map((slug) => ({
    ...notebook_link(slug),
    description: `${slug}.ipynb`,
  }))

export const notebook_routes = () => notebook_subpages().map(({ href }) => href)

export const notebook_prev_next = (): LinkItem[] => slugs(`.html`).map(notebook_link)

export const notebook_entries = () => slugs(`.html`).map((slug) => ({ slug }))

export const read_notebook_html = (slug: string) => {
  try {
    return readFileSync(new URL(`${slug}.html`, examples_dir), `utf8`)
  } catch {
    return null
  }
}
