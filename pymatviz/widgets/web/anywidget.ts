// MatterViz AnyWidget Entry Point

import type { AnyModel, Render } from 'anywidget/types'
import { Composition, Structure, Trajectory } from 'matterviz'
import app_css from 'matterviz/app.css?raw'
import type { ThemeType } from 'matterviz/theme'
import { mount, unmount } from 'svelte'
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
    ${
    is_shadow_dom
      ? app_css
      : `div:is(.structure, .trajectory, .composition) { ${app_css} }`
  }
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

  const has_trajectory = get_prop(model, `trajectory`) !== undefined
  const has_structure = get_prop(model, `structure`) !== undefined
  const has_composition = get_prop(model, `composition`) !== undefined
  const has_data_url = get_prop(model, `data_url`) !== undefined

  if (has_trajectory) render_trajectory(props)
  else if (has_structure) render_structure(props)
  else if (has_composition) render_composition(props)
  // TODO both Structure and Trajectory can be rendered from data_urls, need to find a way to distinguish between them (currently just opting for Trajectory)
  else if (has_data_url) render_trajectory(props)
  else throw new Error(`No valid input found for widget`)
}

const render_composition: Render = ({ model, el }) => {
  const props = {
    composition: get_prop(model, `composition`),
    mode: get_prop(model, `mode`),
    show_percentages: get_prop(model, `show_percentages`),
    color_scheme: get_prop(model, `color_scheme`),
    style: get_prop(model, `style`),
  }

  const component = mount(Composition, { target: el, props })
  instances.set(el, component)
}

const render_structure: Render = ({ model, el }) => {
  const props = {
    structure: get_prop(model, `structure`),
    data_url: get_prop(model, `data_url`),
    scene_props: {
      atom_radius: get_prop(model, `atom_radius`),
      show_atoms: get_prop(model, `show_atoms`),
      auto_rotate: get_prop(model, `auto_rotate`),
      same_size_atoms: get_prop(model, `same_size_atoms`),
      show_bonds: get_prop(model, `show_bonds`),
      show_force_vectors: get_prop(model, `show_force_vectors`),
      force_vector_scale: get_prop(model, `force_vector_scale`),
      force_vector_color: get_prop(model, `force_vector_color`),
      bond_thickness: get_prop(model, `bond_thickness`),
      bond_color: get_prop(model, `bond_color`),
      bonding_strategy: get_prop(model, `bonding_strategy`),
    },
    lattice_props: {
      cell_edge_opacity: get_prop(model, `cell_edge_opacity`),
      cell_surface_opacity: get_prop(model, `cell_surface_opacity`),
      cell_edge_color: get_prop(model, `cell_edge_color`),
      cell_surface_color: get_prop(model, `cell_surface_color`),
      cell_line_width: get_prop(model, `cell_line_width`),
      show_vectors: get_prop(model, `show_vectors`),
    },
    // Display options
    show_site_labels: get_prop(model, `show_site_labels`),
    show_image_atoms: get_prop(model, `show_image_atoms`),
    color_scheme: get_prop(model, `color_scheme`),
    background_color: get_prop(model, `background_color`),
    background_opacity: get_prop(model, `background_opacity`),
    // Widget configuration
    show_controls: get_prop(model, `show_controls`),
    enable_info_panel: get_prop(model, `show_info`),
    fullscreen_toggle: get_prop(model, `show_fullscreen_button`),
    allow_file_drop: false, // Disable file drop in notebook context
    png_dpi: get_prop(model, `png_dpi`),
    style: get_prop(model, `style`),
  }

  const component = mount(Structure, { target: el, props })
  instances.set(el, component)
}

const render_trajectory: Render = ({ model, el }) => {
  const props = {
    trajectory: get_prop(model, `trajectory`),
    data_url: get_prop(model, `data_url`),
    current_step_idx: get_prop(model, `current_step_idx`),
    // Layout and display
    layout: get_prop(model, `layout`),
    display_mode: get_prop(model, `display_mode`),
    show_controls: get_prop(model, `show_controls`),
    show_fullscreen_button: get_prop(model, `show_fullscreen_button`),
    auto_play: get_prop(model, `auto_play`),
    // Widget configuration
    allow_file_drop: false, // Disable file drop in notebook context
    structure_props: {
      scene_props: {
        atom_radius: get_prop(model, `atom_radius`),
        show_atoms: get_prop(model, `show_atoms`),
        auto_rotate: get_prop(model, `auto_rotate`),
        same_size_atoms: get_prop(model, `same_size_atoms`),
        show_bonds: get_prop(model, `show_bonds`),
        show_force_vectors: get_prop(model, `show_force_vectors`),
        force_vector_scale: get_prop(model, `force_vector_scale`),
        force_vector_color: get_prop(model, `force_vector_color`),
        bond_thickness: get_prop(model, `bond_thickness`),
        bond_color: get_prop(model, `bond_color`),
        bonding_strategy: get_prop(model, `bonding_strategy`),
      },
      lattice_props: {
        cell_edge_opacity: get_prop(model, `cell_edge_opacity`),
        cell_surface_opacity: get_prop(model, `cell_surface_opacity`),
        cell_edge_color: get_prop(model, `cell_edge_color`),
        cell_surface_color: get_prop(model, `cell_surface_color`),
        cell_line_width: get_prop(model, `cell_line_width`),
        show_vectors: get_prop(model, `show_vectors`),
      },
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
  }

  const component = mount(Trajectory, { target: el, props })
  instances.set(el, component)
}

export default { render }
