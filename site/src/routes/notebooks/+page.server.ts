import { notebook_subpages } from '$lib/server/notebooks'
import type { PageServerLoad } from './$types'

export const load: PageServerLoad = () => ({ subpages: notebook_subpages() })
