"""Synced traits of each MatterViz widget's model keys, generated from the
bundle's widget-model-schema.json by
`uv run tests/widgets/test_js_prop_parity.py`. Don't edit (excluded from
ruff): widget classes override traits that need defaults or validation.
"""

import traitlets as tl

from pymatviz.widgets._traits import optional_trait


class StructureTraits(tl.HasTraits):
    """structure widget model keys."""

    widget_type = tl.Unicode("structure").tag(sync=True)
    active_pane = optional_trait(tl.CaselessStrEnum, values=['controls', 'export', 'flight', 'info'])
    active_volume_id = optional_trait(tl.Unicode)
    atom_radius = optional_trait(tl.Float)
    auto_rotate = optional_trait(tl.Float)
    background_color = optional_trait(tl.Unicode)
    background_opacity = optional_trait(tl.Float)
    bond_color = optional_trait(tl.Unicode)
    bond_thickness = optional_trait(tl.Float)
    bonding_strategy = optional_trait(tl.CaselessStrEnum, values=['electroneg_ratio', 'explicit_only', 'pair_cutoffs'])
    cell_edge_color = optional_trait(tl.Unicode)
    cell_edge_opacity = optional_trait(tl.Float)
    cell_edge_width = optional_trait(tl.Float)
    cell_surface_color = optional_trait(tl.Unicode)
    cell_surface_opacity = optional_trait(tl.Float)
    cell_type = optional_trait(tl.CaselessStrEnum, values=['conventional', 'original', 'primitive'])
    color_scheme = optional_trait(tl.CaselessStrEnum, values=['Alloy', 'Dark Mode', 'Jmol', 'Muted', 'Pastel', 'Vesta'])
    cutaway = optional_trait(tl.Dict)
    data_url = optional_trait(tl.Union, trait_types=[tl.Unicode(), tl.Dict()])
    display_mode = optional_trait(tl.CaselessStrEnum, values=['slice', 'split', 'structure'])
    enable_info_pane = optional_trait(tl.Bool)
    fullscreen_toggle = optional_trait(tl.Bool)
    gizmo = optional_trait(tl.Union, trait_types=[tl.Bool(), tl.Dict()])
    highlighted_sites = optional_trait(tl.List)
    hovered_site_idx = optional_trait(tl.Float)
    isosurface_settings = optional_trait(tl.Dict)
    png_dpi = optional_trait(tl.Float)
    same_size_atoms = optional_trait(tl.Bool)
    selected_sites = optional_trait(tl.List)
    show_atoms = optional_trait(tl.Bool)
    show_bonds = optional_trait(tl.CaselessStrEnum, values=['always', 'crystals', 'molecules', 'never'])
    show_cell_vectors = optional_trait(tl.Bool)
    show_image_atoms = optional_trait(tl.Bool)
    show_site_indices = optional_trait(tl.Bool)
    show_site_labels = optional_trait(tl.Bool)
    slice_settings = optional_trait(tl.Dict)
    structure = optional_trait(tl.Dict)
    structure_string = optional_trait(tl.Union, trait_types=[tl.Unicode(), tl.Dict()])
    vector_color = optional_trait(tl.Unicode)
    vector_configs = optional_trait(tl.Dict)
    vector_normalize = optional_trait(tl.Bool)
    vector_origin_gap = optional_trait(tl.Float)
    vector_scale = optional_trait(tl.Float)
    vector_uniform_thickness = optional_trait(tl.Bool)
    volumetric_data = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])


class TrajectoryTraits(tl.HasTraits):
    """trajectory widget model keys."""

    widget_type = tl.Unicode("trajectory").tag(sync=True)
    active_pane = optional_trait(tl.CaselessStrEnum, values=['controls', 'export', 'flight', 'hotspots', 'info', 'msd', 'rdf', 'spectroscopy', 'structure-id', 'vacf'])
    atom_radius = optional_trait(tl.Float)
    atom_type_mapping = optional_trait(tl.Dict)
    auto_play = optional_trait(tl.Bool)
    auto_rotate = optional_trait(tl.Float)
    background_color = optional_trait(tl.Unicode)
    background_opacity = optional_trait(tl.Float)
    bond_color = optional_trait(tl.Unicode)
    bond_thickness = optional_trait(tl.Float)
    bonding_strategy = optional_trait(tl.CaselessStrEnum, values=['electroneg_ratio', 'explicit_only', 'pair_cutoffs'])
    cell_edge_color = optional_trait(tl.Unicode)
    cell_edge_opacity = optional_trait(tl.Float)
    cell_edge_width = optional_trait(tl.Float)
    cell_surface_color = optional_trait(tl.Unicode)
    cell_surface_opacity = optional_trait(tl.Float)
    color_scheme = optional_trait(tl.CaselessStrEnum, values=['Alloy', 'Dark Mode', 'Jmol', 'Muted', 'Pastel', 'Vesta'])
    current_step_idx = optional_trait(tl.Float)
    cutaway = optional_trait(tl.Dict)
    data_inspector_open = optional_trait(tl.Bool)
    data_url = optional_trait(tl.Union, trait_types=[tl.Unicode(), tl.Dict()])
    display_mode = optional_trait(tl.CaselessStrEnum, values=['auto', 'plot', 'structure', 'structure+plot'])
    distribution_property = optional_trait(tl.Unicode)
    fullscreen_toggle = optional_trait(tl.Bool)
    gizmo = optional_trait(tl.Union, trait_types=[tl.Bool(), tl.Dict()])
    layout = optional_trait(tl.CaselessStrEnum, values=['auto', 'horizontal', 'vertical'])
    plot_type = optional_trait(tl.CaselessStrEnum, values=['distribution', 'time-series'])
    property_labels = optional_trait(tl.Dict)
    same_size_atoms = optional_trait(tl.Bool)
    show_atoms = optional_trait(tl.Bool)
    show_bonds = optional_trait(tl.CaselessStrEnum, values=['always', 'crystals', 'molecules', 'never'])
    show_cell_vectors = optional_trait(tl.Bool)
    show_image_atoms = optional_trait(tl.Bool)
    show_site_indices = optional_trait(tl.Bool)
    show_site_labels = optional_trait(tl.Bool)
    step_labels = optional_trait(tl.Union, trait_types=[tl.Float(), tl.List()])
    trajectory = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    vector_color = optional_trait(tl.Unicode)
    vector_configs = optional_trait(tl.Dict)
    vector_normalize = optional_trait(tl.Bool)
    vector_origin_gap = optional_trait(tl.Float)
    vector_scale = optional_trait(tl.Float)
    vector_uniform_thickness = optional_trait(tl.Bool)


class ScatterPlotTraits(tl.HasTraits):
    """scatter_plot widget model keys."""

    widget_type = tl.Unicode("scatter_plot").tag(sync=True)
    active_point = optional_trait(tl.Dict)
    annotation = optional_trait(tl.Unicode)
    color_bar = optional_trait(tl.Dict)
    color_scale = optional_trait(tl.Union, trait_types=[tl.CaselessStrEnum(['interpolateBlues', 'interpolateBrBG', 'interpolateBuGn', 'interpolateBuPu', 'interpolateCividis', 'interpolateCool', 'interpolateCubehelixDefault', 'interpolateGnBu', 'interpolateGreens', 'interpolateGreys', 'interpolateInferno', 'interpolateMagma', 'interpolateOrRd', 'interpolateOranges', 'interpolatePRGn', 'interpolatePiYG', 'interpolatePlasma', 'interpolatePuBu', 'interpolatePuBuGn', 'interpolatePuOr', 'interpolatePuRd', 'interpolatePurples', 'interpolateRainbow', 'interpolateRdBu', 'interpolateRdGy', 'interpolateRdPu', 'interpolateRdYlBu', 'interpolateRdYlGn', 'interpolateReds', 'interpolateSinebow', 'interpolateSpectral', 'interpolateTurbo', 'interpolateViridis', 'interpolateWarm', 'interpolateYlGn', 'interpolateYlGnBu', 'interpolateYlOrBr', 'interpolateYlOrRd']), tl.Dict()])
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    density_contours = optional_trait(tl.Union, trait_types=[tl.Bool(), tl.Dict()])
    display = optional_trait(tl.Dict)
    error_bands = optional_trait(tl.List)
    fill_regions = optional_trait(tl.List)
    highlighted_ids = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    hover_config = optional_trait(tl.Dict)
    hovered_point = optional_trait(tl.Dict)
    label_placement_config = optional_trait(tl.Dict)
    legend = optional_trait(tl.Dict)
    line_tween = optional_trait(tl.Dict)
    marker_renderer = optional_trait(tl.CaselessStrEnum, values=['auto', 'canvas', 'svg'])
    padding = optional_trait(tl.Dict)
    point_tween = optional_trait(tl.Dict)
    range_padding = optional_trait(tl.Float)
    ref_lines = optional_trait(tl.List)
    selected_point = optional_trait(tl.Dict)
    series = optional_trait(tl.List)
    show_legend = optional_trait(tl.Bool)
    size_legend = optional_trait(tl.Dict)
    size_scale = optional_trait(tl.Dict)
    styles = optional_trait(tl.Dict)
    x2_axis = optional_trait(tl.Dict)
    x2_range = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    x_range = optional_trait(tl.List)
    y2_axis = optional_trait(tl.Dict)
    y2_range = optional_trait(tl.List)
    y_axis = optional_trait(tl.Dict)
    y_range = optional_trait(tl.List)


class ScatterPlot3dTraits(tl.HasTraits):
    """scatter_plot_3d widget model keys."""

    widget_type = tl.Unicode("scatter_plot_3d").tag(sync=True)
    camera_projection = optional_trait(tl.CaselessStrEnum, values=['orthographic', 'perspective'])
    color_scale = optional_trait(tl.Union, trait_types=[tl.CaselessStrEnum(['interpolateBlues', 'interpolateBrBG', 'interpolateBuGn', 'interpolateBuPu', 'interpolateCividis', 'interpolateCool', 'interpolateCubehelixDefault', 'interpolateGnBu', 'interpolateGreens', 'interpolateGreys', 'interpolateInferno', 'interpolateMagma', 'interpolateOrRd', 'interpolateOranges', 'interpolatePRGn', 'interpolatePiYG', 'interpolatePlasma', 'interpolatePuBu', 'interpolatePuBuGn', 'interpolatePuOr', 'interpolatePuRd', 'interpolatePurples', 'interpolateRainbow', 'interpolateRdBu', 'interpolateRdGy', 'interpolateRdPu', 'interpolateRdYlBu', 'interpolateRdYlGn', 'interpolateReds', 'interpolateSinebow', 'interpolateSpectral', 'interpolateTurbo', 'interpolateViridis', 'interpolateWarm', 'interpolateYlGn', 'interpolateYlGnBu', 'interpolateYlOrBr', 'interpolateYlOrRd']), tl.Dict()])
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    display = optional_trait(tl.Dict)
    legend = optional_trait(tl.Dict)
    point_tween = optional_trait(tl.Dict)
    ref_lines = optional_trait(tl.List)
    ref_planes = optional_trait(tl.List)
    series = optional_trait(tl.List)
    show_legend = optional_trait(tl.Bool)
    size_scale = optional_trait(tl.Dict)
    styles = optional_trait(tl.Dict)
    surfaces = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)
    z_axis = optional_trait(tl.Dict)


class BinnedScatterPlotTraits(tl.HasTraits):
    """binned_scatter_plot widget model keys."""

    widget_type = tl.Unicode("binned_scatter_plot").tag(sync=True)
    annotation = optional_trait(tl.Unicode)
    color_bar = optional_trait(tl.Dict)
    density = optional_trait(tl.Dict)
    fullscreen_toggle = optional_trait(tl.Bool)
    marginals = optional_trait(tl.Union, trait_types=[tl.Bool(), tl.CaselessStrEnum(['cdf', 'histogram', 'kde', 'rug']), tl.Dict()])
    overlays = optional_trait(tl.Dict)
    padding = optional_trait(tl.Dict)
    point_labels = optional_trait(tl.Dict)
    range_padding = optional_trait(tl.Float)
    render_mode = optional_trait(tl.CaselessStrEnum, values=['density', 'points'])
    selected_point_id = optional_trait(tl.Union, trait_types=[tl.Float(), tl.Unicode()])
    series = optional_trait(tl.List)
    size_scale = optional_trait(tl.Dict)
    title = optional_trait(tl.Union, trait_types=[tl.Unicode(), tl.Dict()])
    x_axis = optional_trait(tl.Dict)
    x_range = optional_trait(tl.List)
    y_axis = optional_trait(tl.Dict)
    y_range = optional_trait(tl.List)


class BarPlotTraits(tl.HasTraits):
    """bar_plot widget model keys."""

    widget_type = tl.Unicode("bar_plot").tag(sync=True)
    bar = optional_trait(tl.Dict)
    color_scale = optional_trait(tl.Union, trait_types=[tl.CaselessStrEnum(['interpolateBlues', 'interpolateBrBG', 'interpolateBuGn', 'interpolateBuPu', 'interpolateCividis', 'interpolateCool', 'interpolateCubehelixDefault', 'interpolateGnBu', 'interpolateGreens', 'interpolateGreys', 'interpolateInferno', 'interpolateMagma', 'interpolateOrRd', 'interpolateOranges', 'interpolatePRGn', 'interpolatePiYG', 'interpolatePlasma', 'interpolatePuBu', 'interpolatePuBuGn', 'interpolatePuOr', 'interpolatePuRd', 'interpolatePurples', 'interpolateRainbow', 'interpolateRdBu', 'interpolateRdGy', 'interpolateRdPu', 'interpolateRdYlBu', 'interpolateRdYlGn', 'interpolateReds', 'interpolateSinebow', 'interpolateSpectral', 'interpolateTurbo', 'interpolateViridis', 'interpolateWarm', 'interpolateYlGn', 'interpolateYlGnBu', 'interpolateYlOrBr', 'interpolateYlOrRd']), tl.Dict()])
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    display = optional_trait(tl.Dict)
    legend = optional_trait(tl.Dict)
    line = optional_trait(tl.Dict)
    mode = optional_trait(tl.CaselessStrEnum, values=['grouped', 'overlay', 'stacked'])
    orientation = optional_trait(tl.CaselessStrEnum, values=['horizontal', 'vertical'])
    padding = optional_trait(tl.Dict)
    range_padding = optional_trait(tl.Float)
    ref_lines = optional_trait(tl.List)
    series = optional_trait(tl.List)
    show_legend = optional_trait(tl.Bool)
    size_scale = optional_trait(tl.Dict)
    x2_axis = optional_trait(tl.Dict)
    x2_range = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    x_range = optional_trait(tl.List)
    y2_axis = optional_trait(tl.Dict)
    y2_range = optional_trait(tl.List)
    y_axis = optional_trait(tl.Dict)
    y_range = optional_trait(tl.List)


class HistogramTraits(tl.HasTraits):
    """histogram widget model keys."""

    widget_type = tl.Unicode("histogram").tag(sync=True)
    bar = optional_trait(tl.Dict)
    bins = optional_trait(tl.Float)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    display = optional_trait(tl.Dict)
    legend = optional_trait(tl.Dict)
    mode = optional_trait(tl.CaselessStrEnum, values=['overlay', 'single'])
    padding = optional_trait(tl.Dict)
    range_padding = optional_trait(tl.Float)
    ref_lines = optional_trait(tl.List)
    selected_series_idx = optional_trait(tl.Float)
    series = optional_trait(tl.List)
    show_legend = optional_trait(tl.Bool)
    x2_axis = optional_trait(tl.Dict)
    x2_range = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    x_range = optional_trait(tl.List)
    y2_axis = optional_trait(tl.Dict)
    y2_range = optional_trait(tl.List)
    y_axis = optional_trait(tl.Dict)
    y_range = optional_trait(tl.List)


class CompositionTraits(tl.HasTraits):
    """composition widget model keys."""

    widget_type = tl.Unicode("composition").tag(sync=True)
    color_scheme = optional_trait(tl.CaselessStrEnum, values=['Alloy', 'Dark Mode', 'Jmol', 'Muted', 'Pastel', 'Vesta'])
    composition = optional_trait(tl.Union, trait_types=[tl.Unicode(), tl.Dict()])
    mode = optional_trait(tl.CaselessStrEnum, values=['bar', 'bubble', 'pie'])
    patterns = optional_trait(tl.Dict)
    show_percentages = optional_trait(tl.Bool)


class ConvexHullTraits(tl.HasTraits):
    """convex_hull widget model keys."""

    widget_type = tl.Unicode("convex_hull").tag(sync=True)
    entries = optional_trait(tl.List)
    hull_face_opacity = optional_trait(tl.Float)
    max_hull_dist_show_labels = optional_trait(tl.Float)
    max_hull_dist_show_phases = optional_trait(tl.Float)
    show_hull_faces = optional_trait(tl.Bool)
    show_stable = optional_trait(tl.Bool)
    show_stable_labels = optional_trait(tl.Bool)
    show_unstable = optional_trait(tl.Bool)
    show_unstable_labels = optional_trait(tl.Bool)
    temperature = optional_trait(tl.Float)


class BandStructureTraits(tl.HasTraits):
    """band_structure widget model keys."""

    widget_type = tl.Unicode("band_structure").tag(sync=True)
    band_structs = optional_trait(tl.Dict)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    fermi_level = optional_trait(tl.Float)
    reference_frequency = optional_trait(tl.Float)
    show_legend = optional_trait(tl.Bool)


class DosTraits(tl.HasTraits):
    """dos widget model keys."""

    widget_type = tl.Unicode("dos").tag(sync=True)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    doses = optional_trait(tl.Dict)
    normalize = optional_trait(tl.CaselessStrEnum, values=['integral', 'max', 'sum'])
    orientation = optional_trait(tl.CaselessStrEnum, values=['horizontal', 'vertical'])
    show_legend = optional_trait(tl.Bool)
    sigma = optional_trait(tl.Float)
    spin_mode = optional_trait(tl.CaselessStrEnum, values=['down_only', 'mirror', 'overlay', 'up_only'])
    stack = optional_trait(tl.Bool)


class BandsAndDosTraits(tl.HasTraits):
    """bands_and_dos widget model keys."""

    widget_type = tl.Unicode("bands_and_dos").tag(sync=True)
    band_structs = optional_trait(tl.Dict)
    bz_props = optional_trait(tl.Dict)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    doses = optional_trait(tl.Dict)
    fermi_level = optional_trait(tl.Float)
    normalize = optional_trait(tl.CaselessStrEnum, values=['integral', 'max', 'sum'])
    show_legend = optional_trait(tl.Bool)
    sigma = optional_trait(tl.Float)
    spin_mode = optional_trait(tl.CaselessStrEnum, values=['down_only', 'mirror', 'overlay', 'up_only'])
    stack = optional_trait(tl.Bool)
    structure = optional_trait(tl.Dict)


class FermiSurfaceTraits(tl.HasTraits):
    """fermi_surface widget model keys."""

    widget_type = tl.Unicode("fermi_surface").tag(sync=True)
    band_data = optional_trait(tl.Dict)
    bz_opacity = optional_trait(tl.Float)
    camera_projection = optional_trait(tl.CaselessStrEnum, values=['orthographic', 'perspective'])
    color_bar = optional_trait(tl.Dict)
    color_property = optional_trait(tl.CaselessStrEnum, values=['band', 'property', 'spin'])
    color_scale = optional_trait(tl.Union, trait_types=[tl.CaselessStrEnum(['interpolateBlues', 'interpolateBrBG', 'interpolateBuGn', 'interpolateBuPu', 'interpolateCividis', 'interpolateCool', 'interpolateCubehelixDefault', 'interpolateGnBu', 'interpolateGreens', 'interpolateGreys', 'interpolateInferno', 'interpolateMagma', 'interpolateOrRd', 'interpolateOranges', 'interpolatePRGn', 'interpolatePiYG', 'interpolatePlasma', 'interpolatePuBu', 'interpolatePuBuGn', 'interpolatePuOr', 'interpolatePuRd', 'interpolatePurples', 'interpolateRainbow', 'interpolateRdBu', 'interpolateRdGy', 'interpolateRdPu', 'interpolateRdYlBu', 'interpolateRdYlGn', 'interpolateReds', 'interpolateSinebow', 'interpolateSpectral', 'interpolateTurbo', 'interpolateViridis', 'interpolateWarm', 'interpolateYlGn', 'interpolateYlGnBu', 'interpolateYlOrBr', 'interpolateYlOrRd']), tl.Dict()])
    custom_property_label = optional_trait(tl.Unicode)
    fermi_data = optional_trait(tl.Dict)
    mu = optional_trait(tl.Float)
    property_name = optional_trait(tl.Unicode)
    representation = optional_trait(tl.CaselessStrEnum, values=['solid', 'transparent', 'wireframe'])
    show_bz = optional_trait(tl.Bool)
    show_vectors = optional_trait(tl.Bool)
    surface_opacity = optional_trait(tl.Float)


class BrillouinZoneTraits(tl.HasTraits):
    """brillouin_zone widget model keys."""

    widget_type = tl.Unicode("brillouin_zone").tag(sync=True)
    bz_data = optional_trait(tl.Dict)
    camera_projection = optional_trait(tl.CaselessStrEnum, values=['orthographic', 'perspective'])
    default_k_path = optional_trait(tl.Bool)
    edge_color = optional_trait(tl.Unicode)
    edge_width = optional_trait(tl.Float)
    ibz_color = optional_trait(tl.Unicode)
    ibz_opacity = optional_trait(tl.Float)
    k_path = optional_trait(tl.Dict)
    show_ibz = optional_trait(tl.Bool)
    show_vectors = optional_trait(tl.Bool)
    structure = optional_trait(tl.Dict)
    surface_color = optional_trait(tl.Unicode)
    surface_opacity = optional_trait(tl.Float)


class PhaseDiagramTraits(tl.HasTraits):
    """phase_diagram widget model keys."""

    widget_type = tl.Unicode("phase_diagram").tag(sync=True)
    data = optional_trait(tl.Dict)


class XrdTraits(tl.HasTraits):
    """xrd widget model keys."""

    widget_type = tl.Unicode("xrd").tag(sync=True)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    patterns = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])


class PeriodicTableTraits(tl.HasTraits):
    """periodic_table widget model keys."""

    widget_type = tl.Unicode("periodic_table").tag(sync=True)
    active_elements = optional_trait(tl.List)
    color_bar = optional_trait(tl.Dict)
    color_overrides = optional_trait(tl.Dict)
    color_scale = optional_trait(tl.Union, trait_types=[tl.CaselessStrEnum(['interpolateBlues', 'interpolateBrBG', 'interpolateBuGn', 'interpolateBuPu', 'interpolateCividis', 'interpolateCool', 'interpolateCubehelixDefault', 'interpolateGnBu', 'interpolateGreens', 'interpolateGreys', 'interpolateInferno', 'interpolateMagma', 'interpolateOrRd', 'interpolateOranges', 'interpolatePRGn', 'interpolatePiYG', 'interpolatePlasma', 'interpolatePuBu', 'interpolatePuBuGn', 'interpolatePuOr', 'interpolatePuRd', 'interpolatePurples', 'interpolateRainbow', 'interpolateRdBu', 'interpolateRdGy', 'interpolateRdPu', 'interpolateRdYlBu', 'interpolateRdYlGn', 'interpolateReds', 'interpolateSinebow', 'interpolateSpectral', 'interpolateTurbo', 'interpolateViridis', 'interpolateWarm', 'interpolateYlGn', 'interpolateYlGnBu', 'interpolateYlOrBr', 'interpolateYlOrRd']), tl.Dict()])
    gap = optional_trait(tl.Unicode)
    heatmap_values = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    labels = optional_trait(tl.Dict)
    missing = optional_trait(tl.Dict)
    split_layout = optional_trait(tl.CaselessStrEnum, values=['diagonal', 'horizontal', 'quadrant', 'triangular', 'vertical'])
    tile_plot_config = optional_trait(tl.Dict)
    tile_plots = optional_trait(tl.Dict)
    tile_props = optional_trait(tl.Dict)


class RdfPlotTraits(tl.HasTraits):
    """rdf_plot widget model keys."""

    widget_type = tl.Unicode("rdf_plot").tag(sync=True)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    cutoff = optional_trait(tl.Float)
    mode = optional_trait(tl.CaselessStrEnum, values=['element_pairs', 'full'])
    n_bins = optional_trait(tl.Float)
    patterns = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    show_reference_line = optional_trait(tl.Bool)
    structures = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)


class CoordinationBarPlotTraits(tl.HasTraits):
    """coordination_bar_plot widget model keys."""

    widget_type = tl.Unicode("coordination_bar_plot").tag(sync=True)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    display = optional_trait(tl.Dict)
    legend = optional_trait(tl.Dict)
    mode = optional_trait(tl.CaselessStrEnum, values=['grouped', 'overlay', 'stacked'])
    orientation = optional_trait(tl.CaselessStrEnum, values=['horizontal', 'vertical'])
    padding = optional_trait(tl.Dict)
    show_legend = optional_trait(tl.Bool)
    split_mode = optional_trait(tl.CaselessStrEnum, values=['by_element', 'by_structure', 'none'])
    strategy = optional_trait(tl.CaselessStrEnum, values=['electroneg_ratio', 'explicit_only', 'pair_cutoffs'])
    structures = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)


class HeatmapMatrixTraits(tl.HasTraits):
    """heatmap_matrix widget model keys."""

    widget_type = tl.Unicode("heatmap_matrix").tag(sync=True)
    color_bar = optional_trait(tl.Dict)
    color_scale = optional_trait(tl.Union, trait_types=[tl.CaselessStrEnum(['interpolateBlues', 'interpolateBrBG', 'interpolateBuGn', 'interpolateBuPu', 'interpolateCividis', 'interpolateCool', 'interpolateCubehelixDefault', 'interpolateGnBu', 'interpolateGreens', 'interpolateGreys', 'interpolateInferno', 'interpolateMagma', 'interpolateOrRd', 'interpolateOranges', 'interpolatePRGn', 'interpolatePiYG', 'interpolatePlasma', 'interpolatePuBu', 'interpolatePuBuGn', 'interpolatePuOr', 'interpolatePuRd', 'interpolatePurples', 'interpolateRainbow', 'interpolateRdBu', 'interpolateRdGy', 'interpolateRdPu', 'interpolateRdYlBu', 'interpolateRdYlGn', 'interpolateReds', 'interpolateSinebow', 'interpolateSpectral', 'interpolateTurbo', 'interpolateViridis', 'interpolateWarm', 'interpolateYlGn', 'interpolateYlGnBu', 'interpolateYlOrBr', 'interpolateYlOrRd']), tl.Dict()])
    gap = optional_trait(tl.Unicode)
    label_style = optional_trait(tl.Unicode)
    missing = optional_trait(tl.Dict)
    show_values = optional_trait(tl.Union, trait_types=[tl.Bool(), tl.Unicode()])
    tile_size = optional_trait(tl.Unicode)
    values = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    x_axis = optional_trait(tl.Dict)
    x_items = optional_trait(tl.List)
    y_axis = optional_trait(tl.Dict)
    y_items = optional_trait(tl.List)


class SpacegroupBarTraits(tl.HasTraits):
    """spacegroup_bar widget model keys."""

    widget_type = tl.Unicode("spacegroup_bar").tag(sync=True)
    data = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    orientation = optional_trait(tl.CaselessStrEnum, values=['horizontal', 'vertical'])
    show_counts = optional_trait(tl.Bool)
    show_legend = optional_trait(tl.Bool)
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)


class ChemPotDiagramTraits(tl.HasTraits):
    """chem_pot_diagram widget model keys."""

    widget_type = tl.Unicode("chem_pot_diagram").tag(sync=True)
    config = optional_trait(tl.Dict)
    entries = optional_trait(tl.List)
    height = optional_trait(tl.Float)
    temperature = optional_trait(tl.Float)
    width = optional_trait(tl.Float)


class TreemapTraits(tl.HasTraits):
    """treemap widget model keys."""

    widget_type = tl.Unicode("treemap").tag(sync=True)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    data = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    export_filename = optional_trait(tl.Unicode)
    export_formats = optional_trait(tl.List)
    fullscreen_toggle = optional_trait(tl.Bool)
    label_align = optional_trait(tl.CaselessStrEnum, values=['center', 'top-left'])
    label_fit = optional_trait(tl.CaselessStrEnum, values=['clip', 'hide', 'shrink'])
    label_max_font_size = optional_trait(tl.Float)
    label_min_font_size = optional_trait(tl.Float)
    label_text = optional_trait(tl.CaselessStrEnum, values=['label', 'label+parent-percent', 'label+percent', 'label+value', 'percent', 'value'])
    legend = optional_trait(tl.Dict)
    level_lighten = optional_trait(tl.Float)
    max_children = optional_trait(tl.Float)
    max_depth = optional_trait(tl.Float)
    min_fraction = optional_trait(tl.Float)
    other_label = optional_trait(tl.Unicode)
    padding = optional_trait(tl.Dict)
    padding_inner = optional_trait(tl.Float)
    padding_outer = optional_trait(tl.Float)
    padding_top = optional_trait(tl.Float)
    parent_label_font_size = optional_trait(tl.Float)
    show_breadcrumbs = optional_trait(tl.Bool)
    show_labels = optional_trait(tl.Bool)
    show_legend = optional_trait(tl.Bool)
    sort = optional_trait(tl.CaselessStrEnum, values=['ascending', 'descending', 'none'])
    value_format = optional_trait(tl.Unicode)
    value_mode = optional_trait(tl.CaselessStrEnum, values=['leaf-sum', 'remainder', 'total'])
    zoom_on_click = optional_trait(tl.Bool)
    zoom_root_id = optional_trait(tl.Union, trait_types=[tl.Float(), tl.Unicode()])


class SunburstTraits(tl.HasTraits):
    """sunburst widget model keys."""

    widget_type = tl.Unicode("sunburst").tag(sync=True)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    data = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    export_filename = optional_trait(tl.Unicode)
    export_formats = optional_trait(tl.List)
    fullscreen_toggle = optional_trait(tl.Bool)
    inner_radius = optional_trait(tl.Float)
    label_rotation = optional_trait(tl.CaselessStrEnum, values=['auto', 'horizontal', 'radial', 'tangential'])
    label_text = optional_trait(tl.CaselessStrEnum, values=['label', 'label+parent-percent', 'label+percent', 'label+value', 'percent', 'value'])
    legend = optional_trait(tl.Dict)
    level_lighten = optional_trait(tl.Float)
    max_children = optional_trait(tl.Float)
    max_depth = optional_trait(tl.Float)
    min_fraction = optional_trait(tl.Float)
    other_label = optional_trait(tl.Unicode)
    pad_angle = optional_trait(tl.Float)
    padding = optional_trait(tl.Dict)
    shape = optional_trait(tl.CaselessStrEnum, values=['icicle', 'sunburst'])
    show_breadcrumbs = optional_trait(tl.Bool)
    show_labels = optional_trait(tl.Bool)
    show_legend = optional_trait(tl.Bool)
    sort = optional_trait(tl.CaselessStrEnum, values=['ascending', 'descending', 'none'])
    value_format = optional_trait(tl.Unicode)
    value_mode = optional_trait(tl.CaselessStrEnum, values=['leaf-sum', 'remainder', 'total'])
    zoom_on_click = optional_trait(tl.Bool)
    zoom_root_id = optional_trait(tl.Union, trait_types=[tl.Float(), tl.Unicode()])


class SankeyTraits(tl.HasTraits):
    """sankey widget model keys."""

    widget_type = tl.Unicode("sankey").tag(sync=True)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    data = optional_trait(tl.Dict)
    fullscreen_toggle = optional_trait(tl.Bool)
    iterations = optional_trait(tl.Float)
    legend = optional_trait(tl.Dict)
    link_color_mode = optional_trait(tl.CaselessStrEnum, values=['gradient', 'source', 'static', 'target'])
    link_opacity = optional_trait(tl.Float)
    max_links = optional_trait(tl.Float)
    min_fraction = optional_trait(tl.Float)
    node_align = optional_trait(tl.CaselessStrEnum, values=['center', 'justify', 'left', 'right'])
    node_padding = optional_trait(tl.Float)
    node_width = optional_trait(tl.Float)
    orientation = optional_trait(tl.CaselessStrEnum, values=['horizontal', 'vertical'])
    padding = optional_trait(tl.Dict)
    show_legend = optional_trait(tl.Bool)
    show_node_labels = optional_trait(tl.Bool)
    value_format = optional_trait(tl.Unicode)


class BoxPlotTraits(tl.HasTraits):
    """box_plot widget model keys."""

    widget_type = tl.Unicode("box_plot").tag(sync=True)
    bandwidth = optional_trait(tl.Union, trait_types=[tl.Float(), tl.CaselessStrEnum(['scott', 'silverman'])])
    box = optional_trait(tl.Dict)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    display = optional_trait(tl.Dict)
    kde_clip = optional_trait(tl.List)
    kind = optional_trait(tl.CaselessStrEnum, values=['box', 'violin', 'violin+box'])
    legend = optional_trait(tl.Dict)
    median_style = optional_trait(tl.Dict)
    orientation = optional_trait(tl.CaselessStrEnum, values=['horizontal', 'vertical'])
    padding = optional_trait(tl.Dict)
    point_side = optional_trait(tl.CaselessStrEnum, values=['both', 'negative', 'positive'])
    points = optional_trait(tl.CaselessStrEnum, values=['none', 'strip', 'swarm'])
    range_padding = optional_trait(tl.Float)
    ref_lines = optional_trait(tl.List)
    sample_style = optional_trait(tl.Dict)
    series = optional_trait(tl.List)
    show_legend = optional_trait(tl.Bool)
    show_mean = optional_trait(tl.Bool)
    show_outliers = optional_trait(tl.Bool)
    show_value_labels = optional_trait(tl.Bool)
    side = optional_trait(tl.CaselessStrEnum, values=['both', 'negative', 'positive'])
    title = optional_trait(tl.Union, trait_types=[tl.Unicode(), tl.Dict()])
    value_label_format = optional_trait(tl.Unicode)
    value_label_stat = optional_trait(tl.CaselessStrEnum, values=['mean', 'median'])
    violin_style = optional_trait(tl.Dict)
    violin_width = optional_trait(tl.Float)
    whisker = optional_trait(tl.Dict)
    whisker_mode = optional_trait(tl.CaselessStrEnum, values=['minmax', 'percentile', 'std', 'tukey'])
    whisker_percentiles = optional_trait(tl.List)
    whisker_range = optional_trait(tl.Float)
    x2_axis = optional_trait(tl.Dict)
    x2_range = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    x_range = optional_trait(tl.List)
    y2_axis = optional_trait(tl.Dict)
    y2_range = optional_trait(tl.List)
    y_axis = optional_trait(tl.Dict)
    y_range = optional_trait(tl.List)


class TernaryPlotTraits(tl.HasTraits):
    """ternary_plot widget model keys."""

    widget_type = tl.Unicode("ternary_plot").tag(sync=True)
    color_bar = optional_trait(tl.Dict)
    color_scale = optional_trait(tl.Union, trait_types=[tl.CaselessStrEnum(['interpolateBlues', 'interpolateBrBG', 'interpolateBuGn', 'interpolateBuPu', 'interpolateCividis', 'interpolateCool', 'interpolateCubehelixDefault', 'interpolateGnBu', 'interpolateGreens', 'interpolateGreys', 'interpolateInferno', 'interpolateMagma', 'interpolateOrRd', 'interpolateOranges', 'interpolatePRGn', 'interpolatePiYG', 'interpolatePlasma', 'interpolatePuBu', 'interpolatePuBuGn', 'interpolatePuOr', 'interpolatePuRd', 'interpolatePurples', 'interpolateRainbow', 'interpolateRdBu', 'interpolateRdGy', 'interpolateRdPu', 'interpolateRdYlBu', 'interpolateRdYlGn', 'interpolateReds', 'interpolateSinebow', 'interpolateSpectral', 'interpolateTurbo', 'interpolateViridis', 'interpolateWarm', 'interpolateYlGn', 'interpolateYlGnBu', 'interpolateYlOrBr', 'interpolateYlOrRd']), tl.Dict()])
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    fullscreen_toggle = optional_trait(tl.Bool)
    grid_step = optional_trait(tl.Float)
    labels = optional_trait(tl.List)
    legend = optional_trait(tl.Dict)
    padding = optional_trait(tl.Dict)
    series = optional_trait(tl.List)
    show_grid = optional_trait(tl.Bool)
    show_legend = optional_trait(tl.Bool)
    show_ticks = optional_trait(tl.Bool)
    tick_format = optional_trait(tl.Unicode)


class TernaryPhaseDiagramTraits(tl.HasTraits):
    """ternary_phase_diagram widget model keys."""

    widget_type = tl.Unicode("ternary_phase_diagram").tag(sync=True)
    display = optional_trait(tl.Dict)
    entries = optional_trait(tl.List)
    free_energy_mode = optional_trait(tl.CaselessStrEnum, values=['auto', 'sisso', 'static', 'tabulated'])
    fullscreen_toggle = optional_trait(tl.Bool)
    gas_enabled = optional_trait(tl.Bool)
    gas_pressures = optional_trait(tl.Dict)
    n_samples = optional_trait(tl.Float)
    options = optional_trait(tl.Dict)
    pinned_composition = optional_trait(tl.List)
    selected_phase = optional_trait(tl.Float)
    t_range = optional_trait(tl.List)
    temperature = optional_trait(tl.Float)
    title = optional_trait(tl.Unicode)


class NebPlotTraits(tl.HasTraits):
    """neb_plot widget model keys."""

    widget_type = tl.Unicode("neb_plot").tag(sync=True)
    active_image_idx = optional_trait(tl.Float)
    active_path_key = optional_trait(tl.Unicode)
    annotate_barrier = optional_trait(tl.Bool)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    coord_mode = optional_trait(tl.CaselessStrEnum, values=['arc_length', 'image_index'])
    coord_options = optional_trait(tl.Dict)
    energy_reference = optional_trait(tl.CaselessStrEnum, values=['absolute', 'initial'])
    paths = optional_trait(tl.Union, trait_types=[tl.Dict(), tl.List()])
    show_spline = optional_trait(tl.Bool)
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)


class EosPlotTraits(tl.HasTraits):
    """eos_plot widget model keys."""

    widget_type = tl.Unicode("eos_plot").tag(sync=True)
    controls_open = optional_trait(tl.Bool)
    controls_pane_props = optional_trait(tl.Dict)
    controls_toggle_props = optional_trait(tl.Dict)
    data_label = optional_trait(tl.Unicode)
    energies = optional_trait(tl.List)
    fits = optional_trait(tl.List)
    kinds = optional_trait(tl.List)
    show_fit_params = optional_trait(tl.Bool)
    volumes = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)


class PhononModeExplorerTraits(tl.HasTraits):
    """phonon_mode_explorer widget model keys."""

    widget_type = tl.Unicode("phonon_mode_explorer").tag(sync=True)
    amplitude = optional_trait(tl.Float)
    auto_play = optional_trait(tl.Bool)
    band_coloring = optional_trait(tl.Unicode)
    fps = optional_trait(tl.Float)
    mode_values = optional_trait(tl.Dict)
    n_frames = optional_trait(tl.Float)
    pane_ratio = optional_trait(tl.Float)
    phonons = optional_trait(tl.Dict)
    selection = optional_trait(tl.Dict)
    show_bz_inset = optional_trait(tl.Bool)
    show_vectors = optional_trait(tl.Bool)
    spectrum_kind = optional_trait(tl.CaselessStrEnum, values=['ir', 'raman'])
    supercell = optional_trait(tl.List)
    vector_color = optional_trait(tl.Unicode)
    vector_thickness = optional_trait(tl.Float)
