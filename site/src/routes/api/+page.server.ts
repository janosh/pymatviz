import { readdirSync, readFileSync } from 'node:fs'
import rehypeStarryNight from 'rehype-starry-night'
import rehypeStringify from 'rehype-stringify'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'

const api_docs_dir = new URL(`api-docs/`, `file://${process.cwd()}/`)

export async function load() {
  const markdown_files = readdirSync(api_docs_dir)
    .filter((file_name) => file_name.endsWith(`.md`))
    .toSorted()

  const html = await Promise.all(
    markdown_files.map(async (file_name) => {
      const content = readFileSync(new URL(file_name, api_docs_dir), `utf8`)
      const result = await unified()
        .use(remarkParse)
        .use(remarkRehype)
        .use(rehypeStarryNight)
        .use(rehypeStringify)
        .process(content)

      return String(result).replaceAll(/href="(?!https?:|#|mailto:)[^"]+"/g, `href="#"`)
    }),
  )

  return {
    html: html.join(``),
  }
}
