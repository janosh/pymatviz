// MatterViz AnyWidget Entry Point

import type { AnyModel, Render } from 'anywidget/types'
import {
  Bands,
  BandsAndDos,
  BrillouinZone,
  Composition,
  ConvexHull,
  Dos,
  FermiSurface,
  IsobaricBinaryPhaseDiagram,
  Structure,
  Trajectory,
  XrdPlot,
} from 'matterviz'
import app_css from 'matterviz/app.css?raw'
import type { ThemeType } from 'matterviz/theme'
import { type Component, mount, unmount } from 'svelte'
import {
  detect_parent_theme,
  get_theme_css,
  on_theme_change,
  setup_theme_watchers,
} from './theme-detection'

function inject_app_css(theme_type?: ThemeType, target_element?: HTMLElement): void {
  const style_id = `matterviz-widget-styles`
  const detected_theme = theme_type || detect_parent_theme(target_element)

  // Determine if we're in Shadow DOM (used by marimo cells) and get the appropriate root node
  const root_node = target_element?.getRootNode() || document
  const is_shadow_dom = root_node !== document && root_node instanceof ShadowRoot
  const target_root = is_shadow_dom ? root_node : document

  // Remove existing styles
  const existing_style = is_shadow_dom
    ? target_root.querySelector(`#${style_id}`)
    : document.getElementById(style_id)
  existing_style?.remove()

  // Create style content
  const style_content = `
    ${get_theme_css(detected_theme, is_shadow_dom)}
    .cell-output-ipywidget-background { background: transparent !important; }
    :is(input:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="color"]), textarea, select) {
      background-color: var(--surface-bg); color: var(--text-color); border: 1px solid var(--border-color); border-radius: 4px; padding: 6px 8px;
    }
    :is(input:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="color"]), textarea, select):focus {
      outline: none; border-color: var(--accent-color); box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent-color) 25%, transparent);
    }
    :is(input, textarea)::placeholder { color: var(--text-color-muted); }
    select option { background-color: var(--surface-bg); color: var(--text-color); }
    ${app_css}
  `

  // Apply styles
  if (is_shadow_dom && `adoptedStyleSheets` in target_root) {
    const sheet = new CSSStyleSheet()
    sheet.replaceSync(style_content)
    target_root.adoptedStyleSheets = [...target_root.adoptedStyleSheets, sheet]
    return
  }

  // Fallback: create style element
  const style = document.createElement(`style`)
  style.id = style_id
  style.textContent = style_content
  if (is_shadow_dom) target_root.appendChild(style)
  else document.head.appendChild(style)
}

const instances = new Map<HTMLElement, ReturnType<typeof mount>>()

const get_prop = (model: AnyModel, key: string) => {
  try {
    return model.get(key) ?? undefined
  } catch {
    return undefined
  }
}

// Build an object of { key: model.get(key) } for each key in the list
const pick_props = (model: AnyModel, keys: string[]) =>
  Object.fromEntries(keys.map((key) => [key, get_prop(model, key)]))

// Mount a Svelte component with auto-included base props (notebook context,
// show_controls, style) and track the instance for cleanup
const mount_widget = (
  model: AnyModel,
  el: HTMLElement,
  component: Component<Record<string, unknown>>,
  props: Record<string, unknown>,
) => {
  const base = {
    allow_file_drop: false,
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  }
  instances.set(el, mount(component, { target: el, props: { ...base, ...props } }))
}

// Build scene/lattice props shared by structure and trajectory renderers
const get_scene_props = (model: AnyModel) => ({
  ...pick_props(model, [
    `atom_radius`,
    `show_atoms`,
    `same_size_atoms`,
    `show_bonds`,
    `show_force_vectors`,
    `force_vector_scale`,
    `force_vector_color`,
    `bond_thickness`,
    `bond_color`,
    `bonding_strategy`,
  ]),
  auto_rotate: get_prop(model, `auto_rotate`) ?? 0.2,
})

const get_lattice_props = (model: AnyModel) =>
  pick_props(model, [
    `cell_edge_opacity`,
    `cell_surface_opacity`,
    `cell_edge_color`,
    `cell_surface_color`,
    `cell_edge_width`,
    `show_cell_vectors`,
  ])

// Detect widget type and render
const render: Render = (props) => {
  const { model, el } = props
  inject_app_css(undefined, el)
  setup_theme_watchers(el)

  // Register theme change callback to update CSS when theme changes
  on_theme_change((theme_type) => inject_app_css(theme_type, el))

  // Clean up existing instance
  const existing = instances.get(el)
  if (existing) {
    unmount(existing)
    instances.delete(el)
  }

  const widget_type = get_prop(model, `widget_type`) as string | undefined

  const renderers: Record<string, Render> = {
    structure: render_structure,
    trajectory: render_trajectory,
    composition: render_composition,
    convex_hull: render_convex_hull,
    band_structure: render_band_structure,
    dos: render_dos,
    bands_and_dos: render_bands_and_dos,
    fermi_surface: render_fermi_surface,
    brillouin_zone: render_brillouin_zone,
    phase_diagram: render_phase_diagram,
    xrd: render_xrd,
  }

  const renderer = widget_type ? renderers[widget_type] : undefined
  if (!renderer) throw new Error(`Unknown or missing widget_type: '${widget_type}'`)
  return renderer(props)
}

// === Widget renderers ===

const render_composition: Render = ({ model, el }) => {
  mount_widget(
    model,
    el,
    Composition,
    pick_props(model, [`composition`, `mode`, `show_percentages`, `color_scheme`]),
  )
}

const render_structure: Render = ({ model, el }) => {
  mount_widget(model, el, Structure, {
    ...pick_props(model, [
      `structure`,
      `data_url`,
      `show_site_labels`,
      `show_image_atoms`,
      `color_scheme`,
      `background_color`,
      `background_opacity`,
      `enable_info_pane`,
      `fullscreen_toggle`,
      `png_dpi`,
    ]),
    scene_props: get_scene_props(model),
    lattice_props: get_lattice_props(model),
  })
}

const render_trajectory: Render = ({ model, el }) => {
  mount_widget(model, el, Trajectory, {
    ...pick_props(model, [
      `trajectory`,
      `data_url`,
      `current_step_idx`,
      `layout`,
      `display_mode`,
      `fullscreen_toggle`,
      `auto_play`,
      `step_labels`,
      `property_labels`,
      `units`,
    ]),
    structure_props: {
      scene_props: get_scene_props(model),
      lattice_props: get_lattice_props(model),
      ...pick_props(model, [
        `show_site_labels`,
        `show_image_atoms`,
        `color_scheme`,
        `background_color`,
        `background_opacity`,
      ]),
      fullscreen_toggle: false,
    },
  })
}

const render_convex_hull: Render = ({ model, el }) => {
  mount_widget(
    model,
    el,
    ConvexHull,
    pick_props(model, [
      `entries`,
      `show_stable`,
      `show_unstable`,
      `show_hull_faces`,
      `hull_face_opacity`,
      `show_stable_labels`,
      `show_unstable_labels`,
      `max_hull_dist_show_labels`,
      `max_hull_dist_show_phases`,
      `temperature`,
    ]),
  )
}

const render_band_structure: Render = ({ model, el }) => {
  mount_widget(model, el, Bands, {
    band_structs: get_prop(model, `band_structure`), // renamed traitlet
    ...pick_props(model, [
      `band_type`,
      `show_legend`,
      `fermi_level`,
      `reference_frequency`,
    ]),
  })
}

const render_dos: Render = ({ model, el }) => {
  mount_widget(model, el, Dos, {
    doses: get_prop(model, `dos`), // renamed traitlet
    ...pick_props(model, [
      `stack`,
      `sigma`,
      `normalize`,
      `orientation`,
      `show_legend`,
      `spin_mode`,
    ]),
  })
}

const render_bands_and_dos: Render = ({ model, el }) => {
  mount_widget(model, el, BandsAndDos, {
    band_structs: get_prop(model, `band_structure`), // renamed traitlet
    doses: get_prop(model, `dos`), // renamed traitlet
  })
}

const render_fermi_surface: Render = ({ model, el }) => {
  mount_widget(
    model,
    el,
    FermiSurface,
    pick_props(model, [
      `fermi_data`,
      `band_data`,
      `mu`,
      `representation`,
      `surface_opacity`,
      `show_bz`,
      `bz_opacity`,
      `show_vectors`,
      `camera_projection`,
    ]),
  )
}

const render_brillouin_zone: Render = ({ model, el }) => {
  mount_widget(
    model,
    el,
    BrillouinZone,
    pick_props(model, [
      `structure`,
      `bz_data`,
      `surface_color`,
      `surface_opacity`,
      `edge_color`,
      `edge_width`,
      `show_vectors`,
      `show_ibz`,
      `ibz_color`,
      `ibz_opacity`,
      `camera_projection`,
    ]),
  )
}

const render_phase_diagram: Render = ({ model, el }) => {
  mount_widget(model, el, IsobaricBinaryPhaseDiagram, pick_props(model, [`data`]))
}

const render_xrd: Render = ({ model, el }) => {
  mount_widget(model, el, XrdPlot, pick_props(model, [`patterns`]))
}

export default { render }
