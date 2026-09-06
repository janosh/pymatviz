import { expect, test, vi } from 'vite-plus/test'
import { load } from '../src/routes/api/+page.server'

vi.mock(`node:fs`, () => ({
  readdirSync: () => [`second.md`, `ignored.txt`, `first.md`],
  readFileSync: (path: URL) =>
    path.pathname.endsWith(`first.md`)
      ? `## First\n\n[local](./details) [web](https://example.org) [anchor](#details)`
      : '## Second\n\n```python\nvalue = {"count": 1}\n```',
}))

test(`API docs retain ordering, safe internal links, and highlighted literal code`, async () => {
  const { html } = await load()
  expect(html).toContain(`<h2 id="first">First`)
  expect(html.indexOf(`<h2 id="first">`)).toBeLessThan(html.indexOf(`<h2 id="second">`))
  expect(html).toContain(`href="#first"`)
  expect(html).toContain(`<a href="#">local</a>`)
  expect(html).toContain(`<a href="https://example.org">web</a>`)
  expect(html).toContain(`<a href="#details">anchor</a>`)
  expect(html).toContain(`<span class="pl-`)
  expect(html).toContain(`&#123;`)
  expect(html).not.toContain(`{@html`)
})
