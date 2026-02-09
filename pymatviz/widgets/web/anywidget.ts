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
    :is(.vscode-dark, .dark-theme, [data-jp-theme-light="false"]) :is(input, textarea, select) {
      background-color: #2d2d2d; color: #ffffff; border: 1px solid #555555; border-radius: 4px; padding: 6px 8px;
    }
    :is(.vscode-dark, .dark-theme, [data-jp-theme-light="false"]) :is(input, textarea, select):focus {
      outline: none; border-color: #007acc; box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.2);
    }
    :is(.vscode-dark, .dark-theme, [data-jp-theme-light="false"]) :is(input, textarea)::placeholder { color: #888888; }
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

// Common props for notebook context
const notebook_common = { allow_file_drop: false }

// Helper: mount a Svelte component and track the instance
const mount_widget = (
  el: HTMLElement,
  component: Component<Record<string, unknown>>,
  props: Record<string, unknown>,
) => {
  instances.set(
    el,
    mount(component, { target: el, props: { ...notebook_common, ...props } }),
  )
}

// Build scene/lattice props shared by structure and trajectory renderers
const get_scene_props = (model: AnyModel) => ({
  atom_radius: get_prop(model, `atom_radius`),
  show_atoms: get_prop(model, `show_atoms`),
  auto_rotate: get_prop(model, `auto_rotate`) ?? 0.2,
  same_size_atoms: get_prop(model, `same_size_atoms`),
  show_bonds: get_prop(model, `show_bonds`),
  show_force_vectors: get_prop(model, `show_force_vectors`),
  force_vector_scale: get_prop(model, `force_vector_scale`),
  force_vector_color: get_prop(model, `force_vector_color`),
  bond_thickness: get_prop(model, `bond_thickness`),
  bond_color: get_prop(model, `bond_color`),
  bonding_strategy: get_prop(model, `bonding_strategy`),
})

const get_lattice_props = (model: AnyModel) => ({
  cell_edge_opacity: get_prop(model, `cell_edge_opacity`),
  cell_surface_opacity: get_prop(model, `cell_surface_opacity`),
  cell_edge_color: get_prop(model, `cell_edge_color`),
  cell_surface_color: get_prop(model, `cell_surface_color`),
  cell_edge_width: get_prop(model, `cell_edge_width`),
  show_cell_vectors: get_prop(model, `show_cell_vectors`),
})

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

  const widget_type = get_prop(model, `widget_type`)

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

  if (!widget_type) throw new Error(`widget_type not set on model`)
  const renderer = renderers[widget_type]
  if (renderer) return renderer(props)
  throw new Error(`Unknown widget_type: '${widget_type}'`)
}

// === Widget renderers ===

const render_composition: Render = ({ model, el }) => {
  mount_widget(el, Composition, {
    composition: get_prop(model, `composition`),
    mode: get_prop(model, `mode`),
    show_percentages: get_prop(model, `show_percentages`),
    color_scheme: get_prop(model, `color_scheme`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

const render_structure: Render = ({ model, el }) => {
  mount_widget(el, Structure, {
    structure: get_prop(model, `structure`),
    data_url: get_prop(model, `data_url`),
    scene_props: get_scene_props(model),
    lattice_props: get_lattice_props(model),
    show_site_labels: get_prop(model, `show_site_labels`),
    show_image_atoms: get_prop(model, `show_image_atoms`),
    color_scheme: get_prop(model, `color_scheme`),
    background_color: get_prop(model, `background_color`),
    background_opacity: get_prop(model, `background_opacity`),
    show_controls: get_prop(model, `show_controls`),
    enable_info_pane: get_prop(model, `enable_info_pane`),
    fullscreen_toggle: get_prop(model, `fullscreen_toggle`),
    png_dpi: get_prop(model, `png_dpi`),
    style: get_prop(model, `style`),
  })
}

const render_trajectory: Render = ({ model, el }) => {
  mount_widget(el, Trajectory, {
    trajectory: get_prop(model, `trajectory`),
    data_url: get_prop(model, `data_url`),
    current_step_idx: get_prop(model, `current_step_idx`),
    layout: get_prop(model, `layout`),
    display_mode: get_prop(model, `display_mode`),
    show_controls: get_prop(model, `show_controls`),
    fullscreen_toggle: get_prop(model, `fullscreen_toggle`),
    auto_play: get_prop(model, `auto_play`),
    structure_props: {
      scene_props: get_scene_props(model),
      lattice_props: get_lattice_props(model),
      show_site_labels: get_prop(model, `show_site_labels`),
      show_image_atoms: get_prop(model, `show_image_atoms`),
      color_scheme: get_prop(model, `color_scheme`),
      background_color: get_prop(model, `background_color`),
      background_opacity: get_prop(model, `background_opacity`),
      fullscreen_toggle: false,
    },
    step_labels: get_prop(model, `step_labels`),
    property_labels: get_prop(model, `property_labels`),
    units: get_prop(model, `units`),
    style: get_prop(model, `style`),
  })
}

const render_convex_hull: Render = ({ model, el }) => {
  mount_widget(el, ConvexHull, {
    entries: get_prop(model, `entries`),
    show_stable: get_prop(model, `show_stable`),
    show_unstable: get_prop(model, `show_unstable`),
    show_hull_faces: get_prop(model, `show_hull_faces`),
    hull_face_opacity: get_prop(model, `hull_face_opacity`),
    show_stable_labels: get_prop(model, `show_stable_labels`),
    show_unstable_labels: get_prop(model, `show_unstable_labels`),
    max_hull_dist_show_labels: get_prop(model, `max_hull_dist_show_labels`),
    max_hull_dist_show_phases: get_prop(model, `max_hull_dist_show_phases`),
    temperature: get_prop(model, `temperature`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

const render_band_structure: Render = ({ model, el }) => {
  mount_widget(el, Bands, {
    // Python traitlet `band_structure` -> Svelte prop `band_structs`
    band_structs: get_prop(model, `band_structure`),
    band_type: get_prop(model, `band_type`),
    show_legend: get_prop(model, `show_legend`),
    fermi_level: get_prop(model, `fermi_level`),
    reference_frequency: get_prop(model, `reference_frequency`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

const render_dos: Render = ({ model, el }) => {
  mount_widget(el, Dos, {
    // Python traitlet `dos` -> Svelte prop `doses`
    doses: get_prop(model, `dos`),
    stack: get_prop(model, `stack`),
    sigma: get_prop(model, `sigma`),
    normalize: get_prop(model, `normalize`),
    orientation: get_prop(model, `orientation`),
    show_legend: get_prop(model, `show_legend`),
    spin_mode: get_prop(model, `spin_mode`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

const render_bands_and_dos: Render = ({ model, el }) => {
  mount_widget(el, BandsAndDos, {
    // Python traitlet `band_structure` -> Svelte prop `band_structs`
    band_structs: get_prop(model, `band_structure`),
    // Python traitlet `dos` -> Svelte prop `doses`
    doses: get_prop(model, `dos`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

const render_fermi_surface: Render = ({ model, el }) => {
  mount_widget(el, FermiSurface, {
    fermi_data: get_prop(model, `fermi_data`),
    band_data: get_prop(model, `band_data`),
    mu: get_prop(model, `mu`),
    representation: get_prop(model, `representation`),
    surface_opacity: get_prop(model, `surface_opacity`),
    show_bz: get_prop(model, `show_bz`),
    bz_opacity: get_prop(model, `bz_opacity`),
    show_vectors: get_prop(model, `show_vectors`),
    camera_projection: get_prop(model, `camera_projection`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

const render_brillouin_zone: Render = ({ model, el }) => {
  mount_widget(el, BrillouinZone, {
    structure: get_prop(model, `structure`),
    bz_data: get_prop(model, `bz_data`),
    surface_color: get_prop(model, `surface_color`),
    surface_opacity: get_prop(model, `surface_opacity`),
    edge_color: get_prop(model, `edge_color`),
    edge_width: get_prop(model, `edge_width`),
    show_vectors: get_prop(model, `show_vectors`),
    show_ibz: get_prop(model, `show_ibz`),
    ibz_color: get_prop(model, `ibz_color`),
    ibz_opacity: get_prop(model, `ibz_opacity`),
    camera_projection: get_prop(model, `camera_projection`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

const render_phase_diagram: Render = ({ model, el }) => {
  mount_widget(el, IsobaricBinaryPhaseDiagram, {
    data: get_prop(model, `data`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

const render_xrd: Render = ({ model, el }) => {
  mount_widget(el, XrdPlot, {
    patterns: get_prop(model, `patterns`),
    show_controls: get_prop(model, `show_controls`),
    style: get_prop(model, `style`),
  })
}

export default { render }
