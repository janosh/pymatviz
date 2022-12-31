import { writeFileSync } from 'fs';
import { marked } from "marked";
import { dirname } from 'path';
import { fileURLToPath } from 'url';
import type { PageServerLoad } from './$types';

const dir_name = dirname(fileURLToPath(import.meta.url))

export const load: PageServerLoad = async () => {
  let html = ''
  for (const [path, file] of Object.entries(
    import.meta.glob(`./*.md`, { as: `raw`, eager: true })
  )) {
    const new_file =file
    .replaceAll(`<b>`, ``)
    .replaceAll(`</b>`, ``)
    .replaceAll(
      `src="https://img.shields.io/badge/-source-cccccc?style=flat-square"`,
      `src="https://img.shields.io/badge/source-blue?style=flat" alt="source link"`
    )
    writeFileSync(
      `${dir_name}/${path}`,new_file
      )
       html += marked(new_file)
  }
  return { html }
}
