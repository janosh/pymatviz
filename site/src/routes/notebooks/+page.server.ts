import { notebook_subpages } from '#lib/server/notebooks.js'
import type { PageServerLoad } from './$types'

export const load: PageServerLoad = () => ({ subpages: notebook_subpages() })
