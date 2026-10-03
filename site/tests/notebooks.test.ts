import { expect, test, vi } from 'vite-plus/test'
import {
  notebook_entries,
  notebook_prev_next,
  notebook_routes,
  notebook_subpages,
  read_notebook_html,
} from '#lib/server/notebooks.js'

vi.mock(`node:fs`, () => ({
  readdirSync: () => [`mp_ptable.ipynb`, `eda.html`, `notes.md`, `eda.ipynb`],
  readFileSync: (path: URL) => {
    if (!path.pathname.endsWith(`/eda.html`)) throw new Error(`ENOENT: ${path.pathname}`)
    return `<h1>EDA</h1>`
  },
}))

test(`notebook helpers derive sorted links, routes, entries and HTML from examples dir`, () => {
  expect(notebook_subpages()).toEqual([
    { href: `/notebooks/eda`, label: `eda`, description: `eda.ipynb` },
    { href: `/notebooks/mp_ptable`, label: `mp ptable`, description: `mp_ptable.ipynb` },
  ])
  expect(notebook_routes()).toEqual([`/notebooks/eda`, `/notebooks/mp_ptable`])
  expect(notebook_prev_next()).toEqual([{ href: `/notebooks/eda`, label: `eda` }])
  expect(notebook_entries()).toEqual([{ slug: `eda` }])
  expect(read_notebook_html(`eda`)).toBe(`<h1>EDA</h1>`)
  expect(read_notebook_html(`missing`)).toBeNull()
})
