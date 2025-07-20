// MatterViz AnyWidget Entry Point

import type { AnyModel } from 'anywidget/types'
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

function inject_app_css(theme_type?: ThemeType): void {
  const style_id = `matterviz-widget-styles`

  document.getElementById(style_id)?.remove() // Remove existing styles

  const style = document.createElement(`style`)
  style.id = style_id

  // Use provided theme or detect parent theme
  const detected_theme = theme_type || detect_parent_theme()

  style.textContent = `
    ${get_theme_css(detected_theme)}

    /* something adds an annoying white background, remove it */
    .cell-output-ipywidget-background {
      background: transparent !important;
      background-color: transparent !important;
    }

    /* scope global styles to matterviz widgets to prevent site styles leaking into notebook styles */
    /* this is brittle, will break should component CSS classes in matterviz change, try to find better solution */
    div:is(.structure, .trajectory, .composition) {
      ${app_css}
    }
  `
  document.head.appendChild(style)
}

const instances = new Map<HTMLElement, ReturnType<typeof mount>>()

function render({ model, el }: { model: AnyModel; el: HTMLElement }): void {
  inject_app_css()
  setup_theme_watchers()

  // Register theme change callback to update CSS when theme changes
  on_theme_change(inject_app_css)

  // Clean up existing instance
  const existing = instances.get(el)
  if (existing) {
    unmount(existing)
    instances.delete(el)
  }

  // Detect widget type and render
  const get_prop = (key: string) => {
    try {
      return model.get(key) ?? null
    } catch {
      return null
    }
  }

  const has_trajectory = get_prop(`trajectory`) !== null
  const has_structure = get_prop(`structure`) !== null
  const has_composition = get_prop(`composition`) !== null
  const has_data_url = get_prop(`data_url`) !== null

  if (has_trajectory) render_trajectory({ model, el })
  else if (has_structure) render_structure({ model, el })
  else if (has_composition) render_composition({ model, el })
  // TODO both Structure and Trajectory can be rendered from data_urls, need to find a way to distinguish between them (currently just opting for Trajectory)
  else if (has_data_url) render_trajectory({ model, el })
  else throw new Error(`No valid input found for widget`)
}

function render_composition(
  { model, el }: { model: AnyModel; el: HTMLElement },
): void {
  const props = {
    composition: model.get(`composition`),
    mode: model.get(`mode`),
    show_percentages: model.get(`show_percentages`),
    color_scheme: model.get(`color_scheme`),
    width: model.get(`width`),
    height: model.get(`height`),
  }

  const component = mount(Composition, { target: el, props })
  instances.set(el, component)
}

function render_structure(
  { model, el }: { model: AnyModel; el: HTMLElement },
): void {
  const props = {
    structure: model.get(`structure`),
    data_url: model.get(`data_url`),
    scene_props: {
      atom_radius: model.get(`atom_radius`),
      show_atoms: model.get(`show_atoms`),
      auto_rotate: model.get(`auto_rotate`),
      same_size_atoms: model.get(`same_size_atoms`),
      show_bonds: model.get(`show_bonds`),
      show_force_vectors: model.get(`show_force_vectors`),
      force_vector_scale: model.get(`force_vector_scale`),
      force_vector_color: model.get(`force_vector_color`),
      bond_thickness: model.get(`bond_thickness`),
      bond_color: model.get(`bond_color`),
      bonding_strategy: model.get(`bonding_strategy`),
    },
    lattice_props: {
      cell_edge_opacity: model.get(`cell_edge_opacity`),
      cell_surface_opacity: model.get(`cell_surface_opacity`),
      cell_edge_color: model.get(`cell_edge_color`),
      cell_surface_color: model.get(`cell_surface_color`),
      cell_line_width: model.get(`cell_line_width`),
      show_vectors: model.get(`show_vectors`),
    },

    // Display options
    show_site_labels: model.get(`show_site_labels`),
    show_image_atoms: model.get(`show_image_atoms`),
    color_scheme: model.get(`color_scheme`),
    background_color: model.get(`background_color`),
    background_opacity: model.get(`background_opacity`),

    // Widget configuration
    width: model.get(`width`),
    height: model.get(`height`),
    show_buttons: model.get(`show_controls`),
    enable_info_panel: model.get(`show_info`),
    fullscreen_toggle: model.get(`show_fullscreen_button`),
    allow_file_drop: false, // Disable file drop in notebook context
    png_dpi: model.get(`png_dpi`),
  }

  const component = mount(Structure, { target: el, props })
  instances.set(el, component)
}

function render_trajectory(
  { model, el }: { model: AnyModel; el: HTMLElement },
): void {
  const props = {
    // Core trajectory data
    trajectory: model.get(`trajectory`),
    data_url: model.get(`data_url`),
    current_step_idx: model.get(`current_step_idx`),

    // Layout and display
    layout: model.get(`layout`),
    display_mode: model.get(`display_mode`),
    show_controls: model.get(`show_controls`),
    show_fullscreen_button: model.get(`show_fullscreen_button`),

    // Widget configuration
    width: model.get(`width`),
    height: model.get(`height`),
    allow_file_drop: false, // Disable file drop in notebook context
    structure_props: {
      scene_props: {
        atom_radius: model.get(`atom_radius`),
        show_atoms: model.get(`show_atoms`),
        auto_rotate: model.get(`auto_rotate`),
        same_size_atoms: model.get(`same_size_atoms`),
        show_bonds: model.get(`show_bonds`),
        show_force_vectors: model.get(`show_force_vectors`),
        force_vector_scale: model.get(`force_vector_scale`),
        force_vector_color: model.get(`force_vector_color`),
        bond_thickness: model.get(`bond_thickness`),
        bond_color: model.get(`bond_color`),
        bonding_strategy: model.get(`bonding_strategy`),
      },
      lattice_props: {
        cell_edge_opacity: model.get(`cell_edge_opacity`),
        cell_surface_opacity: model.get(`cell_surface_opacity`),
        cell_edge_color: model.get(`cell_edge_color`),
        cell_surface_color: model.get(`cell_surface_color`),
        cell_line_width: model.get(`cell_line_width`),
        show_vectors: model.get(`show_vectors`),
      },
      show_site_labels: model.get(`show_site_labels`),
      show_image_atoms: model.get(`show_image_atoms`),
      color_scheme: model.get(`color_scheme`),
      background_color: model.get(`background_color`),
      background_opacity: model.get(`background_opacity`),
      fullscreen_toggle: false,
    },
    step_labels: model.get(`step_labels`),
  }

  const component = mount(Trajectory, { target: el, props })
  instances.set(el, component)
}

export default { render }
