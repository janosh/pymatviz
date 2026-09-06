import { readdirSync, readFileSync } from 'node:fs'
import { default_highlighter } from 'svelte-widgets/highlight'
import { assert_ok, create_markdown, render_markdown } from 'svelte-widgets/markdown'

const engine = create_markdown({ highlight: default_highlighter.highlight })
const api_docs_dir = new URL(`api-docs/`, `file://${process.cwd()}/`)

export async function load() {
  const markdown_files = readdirSync(api_docs_dir)
    .filter((file_name) => file_name.endsWith(`.md`))
    .toSorted()

  const html = await Promise.all(
    markdown_files.map(async (file_name) => {
      const content = readFileSync(new URL(file_name, api_docs_dir), `utf8`)
      const document = assert_ok(
        await engine.parse(content, { filename: file_name, dialect: `markdown` }),
      )
      const result = assert_ok(await render_markdown(document))

      return result.replaceAll(/href="(?!https?:|#|mailto:)[^"]+"/gu, `href="#"`)
    }),
  )

  return {
    html: html.join(``),
  }
}
