// Theme Detection for AnyWidget

import { luminance } from 'matterviz/colors'
import 'matterviz/theme'
import { COLOR_THEMES, type ThemeType } from 'matterviz/theme'
// sets globalThis.MATTERVIZ_THEMES and globalThis.MATTERVIZ_CSS_MAP used below in on_theme_change
import 'matterviz/theme/themes'

let current_theme: ThemeType = `light`
const theme_observers: Set<() => void> = new Set()
let mutation_observer: MutationObserver | null = null
let media_query_listener: MediaQueryList | null = null

export function detect_parent_theme(): ThemeType {
  try {
    // 1. Check for Marimo theme CSS class
    if (document.body.classList.contains(`dark-theme`)) return `dark`
    if (document.body.classList.contains(`light-theme`)) return `light`

    // 2. Check for VSCode theme CSS class (recommended here https://github.com/microsoft/vscode/issues/176698#issuecomment-1468638041)
    if (document.body.classList.contains(`vscode-dark`)) return `dark`
    if (document.body.classList.contains(`vscode-light`)) return `light`

    // 2. Use standard CSS media queries (official web API)
    if (globalThis.matchMedia) {
      const prefers_dark = globalThis.matchMedia(`(prefers-color-scheme: dark)`)
      const prefers_light = globalThis.matchMedia(`(prefers-color-scheme: light)`)

      if (prefers_dark.matches) return `dark`
      if (prefers_light.matches) return `light`
    }

    // 3. Check for Jupyter Lab official theme API
    if (globalThis.jupyterlab?.application?.shell?.dataset?.theme) {
      const jupyter_theme = globalThis.jupyterlab.application.shell.dataset
        .theme as string
      return jupyter_theme.includes(`dark`) ? `dark` : `light`
    }

    // 4. Check for Jupyter theme via official CSS custom properties
    const jupyter_theme = getComputedStyle(document.documentElement)
      .getPropertyValue(`--jp-layout-color0`)
    if (jupyter_theme) {
      // Jupyter uses --jp-layout-color0 as background - official property
      const is_dark = is_dark_color(jupyter_theme)
      if (is_dark !== null) return is_dark ? `dark` : `light`
    }

    // 5. Fallback: analyze computed background colors of key elements
    const { body, documentElement } = document
    const body_bg = getComputedStyle(body).backgroundColor
    const root_bg = getComputedStyle(documentElement).backgroundColor

    for (const bg of [body_bg, root_bg]) {
      const is_dark = is_dark_color(bg)
      if (is_dark !== null) return is_dark ? `dark` : `light`
    }

    // 6. Final fallback to light theme
    return `light`
  } catch (error) {
    console.warn(`Theme detection failed, defaulting to light:`, error)
    return `light`
  }
}

// Robust color analysis helper
function is_dark_color(color: string): boolean | null {
  if (!color || [`transparent`, `initial`, `inherit`].includes(color)) return null

  // Use standard luminance calculation (WCAG guidelines)
  return luminance(color) < 0.5
}

function notify_theme_change(): void {
  const new_theme = detect_parent_theme()
  if (new_theme !== current_theme) {
    current_theme = new_theme
    theme_observers.forEach((callback) => callback())
  }
}

export function setup_theme_watchers(): void {
  if (mutation_observer || media_query_listener) return // Avoid duplicates

  try {
    // Watch for class/attribute changes
    mutation_observer = new MutationObserver((mutations) => {
      if (
        mutations.some((mut) =>
          mut.type === `attributes` &&
          (mut.attributeName === `class` || mut.attributeName === `data-theme`)
        )
      ) setTimeout(notify_theme_change, 10) // Debounce
    })

    const observe_opts = { attributes: true, attributeFilter: [`class`, `data-theme`] }
    mutation_observer.observe(document.body, observe_opts)
    mutation_observer.observe(document.documentElement, observe_opts)

    // Watch system preference changes
    if (globalThis.matchMedia) {
      media_query_listener = globalThis.matchMedia(`(prefers-color-scheme: dark)`)
      media_query_listener.addEventListener(`change`, notify_theme_change)
    }

    current_theme = detect_parent_theme() // Set initial theme
  } catch (error) {
    console.warn(`Failed to setup theme watchers:`, error)
  }
}

export function cleanup_theme_watchers(): void {
  mutation_observer?.disconnect()
  mutation_observer = null

  media_query_listener?.removeEventListener(`change`, notify_theme_change)
  media_query_listener = null

  theme_observers.clear()
}

export function on_theme_change(callback: () => void): () => void {
  theme_observers.add(callback)
  return () => theme_observers.delete(callback)
}

export function get_theme_css(theme_type: ThemeType): string {
  const theme_name = COLOR_THEMES[theme_type]

  // Get theme from global (themes.js sets these)
  const theme = globalThis.MATTERVIZ_THEMES?.[theme_name]
  const css_map = globalThis.MATTERVIZ_CSS_MAP

  if (!theme || !css_map) {
    console.warn(`Theme data not available, skipping theme application`)
    return ``
  }

  const css_vars = Object.entries(theme)
    .map(([key, value]) => css_map[key] ? `${css_map[key]}: ${value};` : ``)
    .filter(Boolean)
    .join(`\n\t`)

  return `:root {\n\t${css_vars}\n}`
}
