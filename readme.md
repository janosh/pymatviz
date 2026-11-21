<h1 align="center">
<img src="https://github.com/janosh/pymatviz/raw/main/site/static/favicon.svg" alt="Logo" height="60px">
<br class="hide-in-docs">
pymatviz
</h1>

<h4 align="center" class="toc-exclude">

A toolkit for visualizations in materials informatics.

[![Tests](https://github.com/janosh/pymatviz/actions/workflows/test.yml/badge.svg)](https://github.com/janosh/pymatviz/actions/workflows/test.yml)
[![This project supports Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/pymatviz?logo=pypi&logoColor=white)](https://pypi.org/project/pymatviz)
[![codecov](https://codecov.io/gh/janosh/pymatviz/graph/badge.svg?token=7BG2TZVOBH)](https://codecov.io/gh/janosh/pymatviz)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pymatviz?logo=icloud&logoColor=white)](https://pypistats.org/packages/pymatviz)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.10456384-blue?logo=Zenodo&logoColor=white)](https://zenodo.org/records/10456384)

</h4>

[fig-icon]: assets/svg/fig-icon.svg "View example code"

<slot name="how-to-cite">

> If you use `pymatviz` in your research, [see how to cite](#how-to-cite-pymatviz). Check out [41 existing papers using `pymatviz`](#papers-using-pymatviz) for inspiration!

</slot>

## Installation

```sh
pip install pymatviz
```

See `pyproject.toml` for available extras like `pip install 'pymatviz[export]'` for static image export.

## API Docs

See the [/api][/api] page.

[/api]: https://janosh.github.io/pymatviz/api

## Usage

See the Jupyter notebooks under [`examples/`](examples) for how to use `pymatviz`. PRs with additional examples are welcome! 🙏

|                                                                           |                                                                                                                                                             |                                   |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| [matbench_dielectric_eda.ipynb](examples/matbench_dielectric_eda.ipynb)   | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_dielectric_eda.ipynb)  | [Launch Codespace][codespace url] |
| [matbench_perovskites_eda.ipynb](examples/matbench_perovskites_eda.ipynb) | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_perovskites_eda.ipynb) | [Launch Codespace][codespace url] |

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg
[codespace url]: https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=340898532

## Periodic Table

See [`pymatviz/ptable/figures.py`](pymatviz/ptable/figures.py). The module supports heatmaps, heatmap splits (multiple values per element), histograms, scatter plots and line plots. All visualizations are interactive [MatterViz](https://github.com/janosh/matterviz) widgets: hovering a tile shows its plot enlarged with axes.

|                                     [`ptable_heatmap(atomic_masses)`](pymatviz/ptable/figures.py#L104)                                     | [`ptable_heatmap(compositions, log=True)`](pymatviz/ptable/figures.py#L104) [![fig-icon]](assets/scripts/ptable/ptable_heatmap.py) |
| :---------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: |
|                                                 ![ptable-heatmap-atomic-mass]                                                  |                                                   ![ptable-heatmap-log]                                                    |
|               [`ptable_hists(data)`](pymatviz/ptable/figures.py#L268) [![fig-icon]](assets/scripts/ptable/ptable_hists.py)                | [`ptable_scatter(data, mode="markers")`](pymatviz/ptable/figures.py#L338) [![fig-icon]](assets/scripts/ptable/ptable_scatter.py) |
|                                                          ![ptable-hists]                                                           |                                                 ![ptable-scatter-markers]                                                  |
| [`ptable_heatmap_splits(2_vals_per_elem)`](pymatviz/ptable/figures.py#L204) [![fig-icon]](assets/scripts/ptable/ptable_heatmap_splits.py) |                            [`ptable_heatmap_splits(3_vals_per_elem)`](pymatviz/ptable/figures.py#L204)                            |
|                                                     ![ptable-heatmap-splits-2]                                                     |                                                 ![ptable-heatmap-splits-3]                                                 |

[ptable-heatmap-log]: assets/svg/ptable-heatmap-log.svg
[ptable-heatmap-atomic-mass]: assets/svg/ptable-heatmap-atomic-mass.svg
[ptable-heatmap-splits-2]: assets/svg/ptable-heatmap-splits-2.svg
[ptable-heatmap-splits-3]: assets/svg/ptable-heatmap-splits-3.svg
[ptable-hists]: assets/svg/ptable-hists.svg
[ptable-scatter-markers]: assets/svg/ptable-scatter-markers.svg

## Phonons

| [`BandStructureWidget(band_structs=bands_dict)`](pymatviz/widgets/band_structure.py) [![fig-icon]](assets/scripts/phonons/phonon_bands.py) |                  [`phonon_dos(doses_dict)`](pymatviz/phonons/figures.py) [![fig-icon]](assets/scripts/phonons/phonon_dos.py)                  |
| :-----------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                   ![phonon-bands]                                                   |                                                                   ![phonon-dos]                                                                    |
|                 [`BandsAndDosWidget(band_structs=bands_dict, doses=doses_dict)`](pymatviz/widgets/bands_and_dos.py)                  | [`BandsAndDosWidget(band_structs=single_bands, doses=single_dos)`](pymatviz/widgets/bands_and_dos.py) [![fig-icon]](assets/scripts/phonons/phonon_bands_and_dos.py) |
|                                           ![phonon-bands-and-dos-mp-2758]                                           |                                                          ![phonon-bands-and-dos-mp-23907]                                                          |

[phonon-bands]: assets/svg/phonon-bands-mp-2758.svg
[phonon-dos]: assets/svg/phonon-dos-mp-2758.svg
[phonon-bands-and-dos-mp-2758]: assets/png/phonon-bands-and-dos-mp-2758.png
[phonon-bands-and-dos-mp-23907]: assets/png/phonon-bands-and-dos-mp-23907.png

### Composition Clustering

| [`cluster_compositions(df, *, composition_col, prop_name, embedding_method, projection, n_components=2)`](pymatviz/cluster/composition/plot.py#L122) [![fig-icon]](assets/scripts/cluster/composition/cluster_compositions_matbench.py) | [`cluster_compositions(df, *, composition_col, prop_name, embedding_method, projection, n_components=3)`](pymatviz/cluster/composition/plot.py#L122) |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                                 ![matbench-perovskites-magpie-pca-2d]                                                                                                 |                                                       ![matbench-perovskites-magpie-tsne-3d]                                                       |

[matbench-perovskites-magpie-pca-2d]: assets/svg/matbench-perovskites-magpie-pca-2d.svg
[matbench-perovskites-magpie-tsne-3d]: assets/png/matbench-perovskites-magpie-tsne-3d.png

Visualize 2D or 3D relationships between compositions and properties using multiple embedding and dimensionality reduction techniques:

Embedding methods: **One-hot** encoding of element fractions, **Magpie** features (elemental properties), **Matscholar** element embeddings, **MEGNet** element embeddings

Dimensionality reduction methods: **PCA** (linear), **t-SNE** (non-linear), **UMAP** (non-linear), **Isomap** (non-linear), **Kernel PCA** (non-linear)

Example usage:

```py
import pandas as pd
import pymatviz as pmv

df_comp = pd.DataFrame(
    {"composition": ["Fe2O3", "Al2O3", "SiO2", "TiO2"], "property": [1.0, 2.0, 3.0, 4.0]}
)
widget = pmv.cluster_compositions(
    df_comp,
    prop_name="property",  # optional property to color points by
    embedding_method="one-hot",  # or "magpie", "matscholar_el", "megnet_el", etc.
    projection="pca",  # or "tsne", "umap", "isomap", "kernel_pca", etc.
    show_chem_sys="shape",  # "color" | "shape" | "color+shape" | None
    n_components=2,  # or 3 for 3D plots
)
widget.show()
```

## Structure Clustering

On the roadmap but no ETA yet.

## Structure

See [`pymatviz/widgets/structure.py`](pymatviz/widgets/structure.py).

|                            [`StructureWidget(structure=hea_structure)`](pymatviz/widgets/structure.py)                            | [`StructureWidget(structure=lco_supercell)`](pymatviz/widgets/structure.py) [![fig-icon]](assets/scripts/structure/structure_widget.py) |
| :------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
|                                                 ![hea-structure-3d]                                                  |                                                  ![lco-structure-3d]                                                   |
| [`StructureWidget(structure=batio3)`](pymatviz/widgets/structure.py) [![fig-icon]](assets/scripts/structure/structure_widget.py) | [`StructureWidget(structure=mos2)`](pymatviz/widgets/structure.py) [![fig-icon]](assets/scripts/structure/structure_widget.py) |
|                                          ![batio3-structure-3d]                                           | ![mos2-structure-3d] |

[batio3-structure-3d]: assets/png/batio3-structure-3d.png
[hea-structure-3d]: assets/png/hea-structure-3d.png
[lco-structure-3d]: assets/png/lco-structure-3d.png
[mos2-structure-3d]: assets/png/mos2-structure-3d.png

## Interactive Widgets

See [`pymatviz/widgets`](pymatviz/widgets). Interactive 3D structure, molecular dynamics trajectory and composition visualization widgets for [Jupyter](https://jupyter.org), [Marimo](https://marimo.io), and VSCode notebooks, powered by [anywidget](https://anywidget.dev) and [MatterViz](https://matterviz.janosh.dev) (<https://github.com/janosh/matterviz>). Supports pymatgen `Structure`, ASE `Atoms`, and `PhonopyAtoms`, as well as ASE, `pymatgen` and plain Python trajectory formats.

```py
from pymatviz import StructureWidget, CompositionWidget, TrajectoryWidget
from pymatgen.core import Structure, Composition

# Interactive 3D structure visualization
structure = Structure.from_file("structure.cif")
struct_widget = StructureWidget(structure=structure)

# Interactive composition visualization
composition = Composition("Fe2O3")
comp_widget = CompositionWidget(composition=composition)

# Interactive trajectory visualization
trajectory1 = [struct1, struct2, struct3]  # List of structures
traj_widget1 = TrajectoryWidget(trajectory=trajectory1)

trajectory2 = [{"structure": struct1, "energy": 1.0}, {"structure": struct2, "energy": 2.0}, {"structure": struct3, "energy": 3.0}]  # dicts with "structure" and property values
traj_widget2 = TrajectoryWidget(trajectory=trajectory2)
```

All widgets support export and inspection APIs that work in plain Python scripts, CI, and agent workflows — no notebook frontend required:

```py
widget.to_html("structure.html")  # standalone interactive HTML page (pan/zoom/rotate work)
widget.to_img("structure.png", dpi=150)  # PNG/JPEG/SVG/PDF via headless Chromium
widget.describe()  # machine-readable facts: counts, ranges, formula, chemical system
widget.render_report()  # headless render health check: blank fraction, warnings
```

Lay out several widgets with `widget_grid({title: widget, ...}, n_cols=3)`, e.g. one `StructureWidget` per structure. It returns an `ipywidgets.GridBox` of titled cells: export each widget from its `.children`.

`to_img` and `render_report` require [Playwright](https://playwright.dev/python): `pip install pymatviz[export] && playwright install chromium` (on Linux CI runners without Chromium system libraries, use `playwright install --with-deps chromium`). Exports render in headless Chromium. Set the `PYMATVIZ_STATIC` env var to make `widget.show()` render static PNGs instead of live widgets (useful for CI notebook execution).

Widget constructor kwargs mirror the props of the matching MatterViz component one-to-one (e.g. `BandStructureWidget(band_structs=...)`, `DosWidget(doses=...)`, `HeatmapMatrixWidget(color_scale={"type": "log"})`), and every plot widget takes `show_controls`, `controls_open`, `controls_toggle_props` and `controls_pane_props` for its control pane. The frontend is the prebuilt [`matterviz-anywidget`](https://www.npmjs.com/package/matterviz-anywidget) bundle selected by `MATTERVIZ_ANYWIDGET_VERSION` in [`pymatviz/widgets/matterviz.py`](pymatviz/widgets/matterviz.py): `"main"` between releases (matterviz main's bundle from `matterviz.janosh.dev/anywidget/`, re-fetched each session), a published npm version in releases (fetched from jsDelivr and cached on disk); set `MATTERVIZ_ANYWIDGET_DIR` to a local `matterviz/extensions/anywidget/build` to develop against an unreleased MatterViz.

**Examples:**

- [Jupyter notebook demo](examples/widgets/jupyter_demo.ipynb)
- [Marimo demo](examples/widgets/marimo_demo.py)
- [VSCode interactive demo](examples/widgets/vscode_interactive_demo.py)

> [!TIP]
> Checkout the **✅ MatterViz VSCode extension** for using the same viewers directly in VSCode/Cursor editor tabs for rendering local and remote files: [marketplace.visualstudio.com/items?itemName=janosh.matterviz](https://marketplace.visualstudio.com/items?itemName=janosh.matterviz)

Importing `pymatviz` auto-registers all widgets for their respective sets of supported objects via [`register_matterviz_widgets()`](pymatviz/widgets/mime.py).

## Brillouin Zone

See [`pymatviz/widgets/brillouin_zone.py`](pymatviz/widgets/brillouin_zone.py). MatterViz standardizes any input cell, so the zone and its high-symmetry k-path labels match the Setyawan-Curtarolo convention.

|   [`BrillouinZoneWidget(structure=cubic_struct)`](pymatviz/widgets/brillouin_zone.py) [![fig-icon]](assets/scripts/brillouin/brillouin_zone_widget.py)    |  [`BrillouinZoneWidget(structure=hexagonal_struct)`](pymatviz/widgets/brillouin_zone.py)   |
| :------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------: |
|                                                   ![brillouin-cubic-mp-10018]                                                    |                   ![brillouin-hexagonal-mp-862690]                    |
| [`BrillouinZoneWidget(structure=monoclinic_struct)`](pymatviz/widgets/brillouin_zone.py) [![fig-icon]](assets/scripts/brillouin/brillouin_zone_widget.py) | [`BrillouinZoneWidget(structure=orthorhombic_struct)`](pymatviz/widgets/brillouin_zone.py) |
|                                                ![brillouin-monoclinic-mp-1183089]                                                |                  ![brillouin-orthorhombic-mp-1183085]                 |

[brillouin-cubic-mp-10018]: assets/png/brillouin-cubic-mp-10018.png
[brillouin-hexagonal-mp-862690]: assets/png/brillouin-hexagonal-mp-862690.png
[brillouin-monoclinic-mp-1183089]: assets/png/brillouin-monoclinic-mp-1183089.png
[brillouin-orthorhombic-mp-1183085]: assets/png/brillouin-orthorhombic-mp-1183085.png

## X-Ray Diffraction

See [`pymatviz/xrd.py`](pymatviz/xrd.py).

|             [`xrd_pattern(pattern)`](pymatviz/xrd.py#L53) [![fig-icon]](assets/scripts/xrd/xrd_pattern.py)             |  [`xrd_pattern({key1: patt1, key2: patt2})`](pymatviz/xrd.py#L53)   |
| :--------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
|                                                     ![xrd-pattern]                                                     |                       ![xrd-pattern-multiple]                       |

[xrd-pattern]: assets/svg/xrd-pattern.svg
[xrd-pattern-multiple]: assets/svg/xrd-pattern-multiple.svg

## Radial Distribution Functions

See [`pymatviz/rdf/figures.py`](pymatviz/rdf/figures.py).

| [`element_pair_rdfs(pmg_struct)`](pymatviz/rdf/figures.py#L74) | [`element_pair_rdfs({"A": struct1, "B": struct2})`](pymatviz/rdf/figures.py#L74) [![fig-icon]](assets/scripts/rdf/element_pair_rdfs.py) |
| :------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
|                 ![element-pair-rdfs-Na8Nb8O24]                 |                                                ![element-pair-rdfs-crystal-vs-amorphous]                                                |

[element-pair-rdfs-Na8Nb8O24]: assets/svg/element-pair-rdfs-Na8Nb8O24.svg
[element-pair-rdfs-crystal-vs-amorphous]: assets/svg/element-pair-rdfs-crystal-vs-amorphous.svg

## Coordination

See [`pymatviz/coordination/figures.py`](pymatviz/coordination/figures.py).

|              [`coordination_hist(struct_dict)`](pymatviz/coordination/figures.py#L73)              |            [`coordination_hist(struct_dict, split_mode="by_element")`](pymatviz/coordination/figures.py#L73) [![fig-icon]](assets/scripts/coordination/coordination_hist.py)             |
| :------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                    ![coordination-hist-single]                                     |                                                                  ![coordination-hist-by-element]                                                                   |
| [`coordination_vs_cutoff_line(struct_dict)`](pymatviz/coordination/figures.py#L176) | [`coordination_vs_cutoff_line(struct_dict)`](pymatviz/coordination/figures.py#L176) [![fig-icon]](assets/scripts/coordination/coordination_vs_cutoff_line.py#L52) |
|                                  ![coordination-vs-cutoff-single]                                  |                                                                        ![coordination-vs-cutoff-multiple]                                                                        |

[coordination-hist-single]: assets/svg/coordination-hist-single.svg
[coordination-hist-by-element]: assets/svg/coordination-hist-by-element.svg
[coordination-vs-cutoff-single]: assets/svg/coordination-vs-cutoff-single.svg
[coordination-vs-cutoff-multiple]: assets/svg/coordination-vs-cutoff-multiple.svg

## Sunburst

`chem_sys_sunburst` and `chem_env_sunburst` return interactive MatterViz `SunburstWidget` instances. Pass the same nested `data` to `SunburstWidget` or `TreemapWidget` to switch views. Use `widget.to_html("hierarchy.html", inline=True)` for a self-contained interactive file (the default `to_html` loads the MatterViz bundle from the network) or `widget.to_img("hierarchy.svg")` for a vector figure.

| [`spacegroup_sunburst([65, 134, 225, ...])`](pymatviz/sunburst/spacegroup.py) [![fig-icon]](assets/scripts/sunburst/spacegroup_sunburst.py) | [`chem_sys_sunburst(["FeO", "Fe2O3", "LiPO4", ...])`](pymatviz/sunburst/chem_sys.py) [![fig-icon]](assets/scripts/sunburst/chem_sys_sunburst.py) |
| :-----------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                             ![spg-num-sunburst]                                                             |                                                          ![chem-sys-sunburst-ward-bmg]                                                           |
|                                     [`chem_env_sunburst(single_struct)`](pymatviz/sunburst/chem_env.py)                                     |                                      [`chem_env_sunburst(structures, normalize=True)`](pymatviz/sunburst/chem_env.py)                                      |
|                                                         ![chem-env-sunburst-basic]                                                          |                                                          ![chem-env-sunburst-normalized]                                                          |

[spg-num-sunburst]: assets/svg/spg-num-sunburst.svg
[chem-sys-sunburst-ward-bmg]: assets/svg/chem-sys-sunburst-ward-bmg.svg
[chem-env-sunburst-basic]: assets/svg/chem-env-sunburst-basic.svg
[chem-env-sunburst-normalized]: assets/svg/chem-env-sunburst-normalized.svg

## Treemap

Chemical-system and coordination-environment treemaps return MatterViz `TreemapWidget` instances. `max_systems` limits systems per arity. Set `overflow="drop"` to omit the remaining systems or use the default `"other"` to retain their combined count.

| [`chem_sys_treemap(formulas, group_by="formula")`](pymatviz/treemap/chem_sys.py#L19) [![fig-icon]](assets/scripts/treemap/chem_sys_treemap.py) | [`chem_sys_treemap(ward_compositions)`](pymatviz/treemap/chem_sys.py#L19) |
| :----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: |
|                                                           ![chem-sys-treemap-formula]                                                            |                                        ![chem-sys-treemap-ward-bmg]                                        |
|           [`chem_env_treemap(structure, chem_env_settings="chemenv")`](pymatviz/treemap/chem_env.py#L18) [![fig-icon]](assets/scripts/treemap/chem_env_treemap.py)           |     [`chem_env_treemap(structures, max_coordination_numbers=3, max_environments=4)`](pymatviz/treemap/chem_env.py#L18)     |
|                                                            ![chem-env-treemap-basic]                                                             |                                     ![chem-env-treemap-large-dataset]                                      |
|             [`py_pkg_treemap("pymatviz")`](pymatviz/treemap/py_pkg.py#L392) [![fig-icon]](assets/scripts/treemap/py_pkg_treemap.py)              |           [`py_pkg_treemap(["pymatviz", "flame", "pymatgen"])`](pymatviz/treemap/py_pkg.py#L392)           |
|                                                            ![py-pkg-treemap-pymatviz]                                                            |                                         ![py-pkg-treemap-multiple]                                         |
|   [`py_pkg_treemap("pymatviz", color_by=coverage_by_path)`](pymatviz/treemap/py_pkg.py#L392) [![fig-icon]](assets/scripts/treemap/py_pkg_treemap.py)   | [`py_pkg_treemap("pymatgen", color_by=coverage_by_path, color_range=(0, 100))`](pymatviz/treemap/py_pkg.py#L392) |
|                                                       ![py-pkg-treemap-pymatviz-coverage]                                                        |                                    ![py-pkg-treemap-pymatgen-coverage]                                     |

For coverage coloring, pass a `color_by` dictionary mapping file paths to coverage percentages. Use `color_range=(0, 100)` for a fixed percentage scale. The gallery script shows how to read a coverage.py JSON report.

[chem-sys-treemap-formula]: assets/svg/chem-sys-treemap-formula.svg
[chem-sys-treemap-ward-bmg]: assets/svg/chem-sys-treemap-ward-bmg.svg
[py-pkg-treemap-pymatviz]: assets/svg/py-pkg-treemap-pymatviz.svg
[py-pkg-treemap-multiple]: assets/svg/py-pkg-treemap-multiple.svg
[py-pkg-treemap-pymatgen-coverage]: assets/svg/py-pkg-treemap-pymatgen-coverage.svg
[py-pkg-treemap-pymatviz-coverage]: assets/svg/py-pkg-treemap-pymatviz-coverage.svg
[chem-env-treemap-large-dataset]: assets/svg/chem-env-treemap-large-dataset.svg
[chem-env-treemap-basic]: assets/svg/chem-env-treemap-basic.svg

## Rainclouds

See [`pymatviz/rainclouds.py`](pymatviz/rainclouds.py).

| [`rainclouds(two_key_dict)`](pymatviz/rainclouds.py#L19) [![fig-icon]](assets/scripts/rainclouds/rainclouds.py) | [`rainclouds(three_key_dict)`](pymatviz/rainclouds.py#L19) |
| :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------: |
|                                              ![rainclouds-bimodal]                                              |                   ![rainclouds-trimodal]                   |

[rainclouds-bimodal]: assets/svg/rainclouds-bimodal.svg
[rainclouds-trimodal]: assets/svg/rainclouds-trimodal.svg

## Sankey

See [`pymatviz/sankey.py`](pymatviz/sankey.py).

| [`sankey(df_space_groups, crystal_system, spg_num)`](pymatviz/sankey.py) [![fig-icon]](assets/scripts/sankey/sankey.py) |
| :---: |
| ![sankey-crystal-sys-to-spg-num] |

[sankey-crystal-sys-to-spg-num]: assets/svg/sankey-crystal-sys-to-spg-num.svg

## Bar Plots

See [`pymatviz/bar.py`](pymatviz/bar.py).

| [`spacegroup_bar([65, 134, 225, ...])`](pymatviz/bar.py#L20) [![fig-icon]](assets/scripts/bar/spacegroup_bar.py) | [`spacegroup_bar(["C2/m", "P-43m", "Fm-3m", ...])`](pymatviz/bar.py#L20) |
| :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------: |
|                                              ![spg-num-bar]                                              |                        ![spg-symbol-bar]                         |

[spg-symbol-bar]: assets/svg/spg-symbol-bar.svg
[spg-num-bar]: assets/svg/spg-num-bar.svg

## Histograms

See [`pymatviz/histogram.py`](pymatviz/histogram.py).

| [`elements_hist(compositions, log_y=True, show_values='count')`](pymatviz/histogram.py#L25) [![fig-icon]](assets/scripts/histogram/elements_hist.py) | [`histogram({'key1': values1, 'key2': values2})`](pymatviz/histogram.py#L94) [![fig-icon]](assets/scripts/histogram/histogram.py) |
| :-----------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: |
|                                                                 ![elements-hist]                                                                  |                                                         ![histogram]                                                         |

[histogram]: assets/svg/histogram.svg

## Scatter Plots

See [`pymatviz/scatter.py`](pymatviz/scatter.py).

| [`density_scatter(xs, ys, ...)`](pymatviz/scatter.py) [![fig-icon]](assets/scripts/scatter/density_scatter.py) | [`density_hexbin(xs, ys, bin_px=1.5, marginal_bins=100)`](pymatviz/scatter.py) [![fig-icon]](assets/scripts/scatter/density_scatter.py) |
| :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
|                                               ![density-scatter]                                               |                                               ![density-scatter-with-hist]                                               |
|  [`density_hexbin(xs, ys, ...)`](pymatviz/scatter.py) [![fig-icon]](assets/scripts/scatter/density_hexbin.py)  |  [`density_hexbin(xs, ys, bin_px=8, marginal_bins=100)`](pymatviz/scatter.py) [![fig-icon]](assets/scripts/scatter/density_hexbin.py)  |
|                                               ![density-hexbin]                                                |                                               ![density-hexbin-with-hist]                                                |

[density-hexbin-with-hist]: assets/svg/density-hexbin-with-hist.svg
[density-hexbin]: assets/svg/density-hexbin.svg
[density-scatter-with-hist]: assets/svg/density-scatter-with-hist.svg
[density-scatter]: assets/svg/density-scatter.svg

## Uncertainty

See [`pymatviz/uncertainty.py`](pymatviz/uncertainty.py).

|             [`qq_gaussian(y_true, y_pred, y_std)`](pymatviz/uncertainty.py#L73) [![fig-icon]](assets/scripts/uncertainty/qq_gaussian.py)              |       [`qq_gaussian(y_true, y_pred, y_std: dict)`](pymatviz/uncertainty.py#L73)        |
| :---------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
|                                                                    ![qq-gaussian]                                                                     |                                ![qq-gaussian-multiple]                                 |
| [`error_decay_with_uncert(y_true, y_pred, y_std)`](pymatviz/uncertainty.py#L160) [![fig-icon]](assets/scripts/uncertainty/error_decay_with_uncert.py) | [`error_decay_with_uncert(y_true, y_pred, y_std: dict)`](pymatviz/uncertainty.py#L160) |
|                                                              ![error-decay-with-uncert]                                                               |                          ![error-decay-with-uncert-multiple]                           |

## Classification

See [`pymatviz/classify/confusion_matrix.py`](pymatviz/classify/confusion_matrix.py).

| [`confusion_matrix(conf_mat, ...)`](pymatviz/classify/confusion_matrix.py#L34) | [`confusion_matrix(y_true, y_pred, ...)`](pymatviz/classify/confusion_matrix.py#L34) [![fig-icon]](assets/scripts/classify/confusion_matrix.py) |
| :----------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
|                         ![stability-confusion-matrix]                          |                                                       ![crystal-system-confusion-matrix]                                                        |

See [`pymatviz/classify/curves.py`](pymatviz/classify/curves.py).

| [`roc_curve(targets, probs_positive)`](pymatviz/classify/curves.py#L144) [![fig-icon]](assets/scripts/classify/roc_curve.py) | [`precision_recall_curve(targets, probs_positive)`](pymatviz/classify/curves.py#L234) [![fig-icon]](assets/scripts/classify/precision_recall_curve.py) |
| :-------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                ![roc-curve-multiple]                                                 |                                                       ![precision-recall-curve-multiple]                                                        |

[roc-curve-multiple]: assets/svg/roc-curve-multiple.svg
[precision-recall-curve-multiple]: assets/svg/precision-recall-curve-multiple.svg
[stability-confusion-matrix]: assets/svg/stability-confusion-matrix.svg
[crystal-system-confusion-matrix]: assets/svg/crystal-system-confusion-matrix.svg
[error-decay-with-uncert-multiple]: assets/svg/error-decay-with-uncert-multiple.svg
[error-decay-with-uncert]: assets/svg/error-decay-with-uncert.svg
[elements-hist]: assets/svg/elements-hist.svg
[qq-gaussian-multiple]: assets/svg/qq-gaussian-multiple.svg
[qq-gaussian]: assets/svg/qq-gaussian.svg

## How to cite `pymatviz`

See [`citation.cff`](citation.cff) or cite the [Zenodo record](https://zenodo.org/badge/latestdoi/340898532) using the following BibTeX entry:

```bib
@software{riebesell_pymatviz_2022,
  title = {Pymatviz: visualization toolkit for materials informatics},
  author = {Riebesell, Janosh and Yang, Haoyu and Goodall, Rhys and Baird, Sterling G.},
  date = {2022-10-01},
  year = {2022},
  doi = {10.5281/zenodo.7486816},
  url = {https://github.com/janosh/pymatviz},
  note = {10.5281/zenodo.7486816 - https://github.com/janosh/pymatviz},
  urldate = {2023-01-01}, % optional, replace with your date of access
  version = {0.8.2}, % replace with the version you use
}
```

## Papers using `pymatviz`

Sorted by number of citations, then year. Last updated 2026-02-25. Auto-generated [from Google Scholar](https://scholar.google.com/scholar?q=pymatviz). Manual additions [via PR](https://github.com/janosh/pymatviz/edit/main/readme.md) welcome.

1. L Barroso-Luque, M Shuaibi, X Fu et al. (2024). [Open materials 2024 (omat24) inorganic materials dataset and models](https://arxiv.org/abs/2410.12771) (cited by 230)
1. C Zeni, R Pinsler, D Zügner et al. (2023). [Mattergen: a generative model for inorganic materials design](https://arxiv.org/abs/2312.03687) (cited by 166)
1. C Chen, DT Nguyen, SJ Lee et al. (2024). [Accelerating computational materials discovery with machine learning and cloud high-performance computing: from large-scale screening to experimental validation](https://pubs.acs.org/doi/abs/10.1021/jacs.4c03849) (cited by 131)
1. J Riebesell, REA Goodall, P Benner et al. (2023). [Matbench Discovery--A framework to evaluate machine learning crystal stability predictions](https://arxiv.org/abs/2308.14920) (cited by 106)
1. H Yu, M Giantomassi, G Materzanini (2024). [Systematic assessment of various universal machine‐learning interatomic potentials](https://onlinelibrary.wiley.com/doi/abs/10.1002/mgea.58) (cited by 69)
1. M Gibaldi, A Kapeliukha, A White et al. (2025). [MOSAEC-DB: a comprehensive database of experimental metal–organic frameworks with verified chemical accuracy suitable for molecular simulations](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc07438f) (cited by 34)
1. AA Naik, C Ertural, N Dhamrait et al. (2023). [A quantum-chemical bonding database for solid-state materials](https://www.nature.com/articles/s41597-023-02477-5) (cited by 28)
1. F Therrien, J Abou Haibeh, D Sharma et al. (2026). [OBELiX: A curated dataset of crystal structures and experimentally measured ionic conductivities for lithium solid-state electrolytes](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00441a) (cited by 9)
1. K Li, AN Rubungo, X Lei et al. (2024). [Probing out-of-distribution generalization in machine learning for materials](https://arxiv.org/abs/2406.06489) (cited by 9)
1. Y Zhou, X He, Z Li (2025). [Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning](https://arxiv.org/abs/2506.10521) (cited by 6)
1. HH Li, Q Chen, G Ceder (2024). [Voltage Mining for (De) lithiation-Stabilized Cathodes and a Machine Learning Model for Li-Ion Cathode Voltage](https://pubs.acs.org/doi/abs/10.1021/acsami.4c15742) (cited by 5)
1. A Peng, X Liu, MY Guo et al. (2025). [The openlam challenges](https://arxiv.org/abs/2501.16358) (cited by 3)
1. J Nam, S Liu, G Winter et al. (2025). [Flow matching for accelerated simulation of atomic transport in crystalline materials](https://www.nature.com/articles/s42256-025-01125-4) (cited by 3)
1. A Onwuli, KT Butler, A Walsh (2024). [Ionic species representations for materials informatics](https://pubs.aip.org/aip/aml/article/2/3/036112/3313198) (cited by 3)
1. N Tuchinda, CA Schuh (2025). [Grain Boundary Segregation and Embrittlement of Aluminum Binary Alloys from First Principles](https://arxiv.org/abs/2502.01579) (cited by 2)
1. R Nduma, H Park, A Walsh (2025). [Crystalyse: a multi-tool agent for materials design](https://arxiv.org/abs/2512.00977) (cited by 2)
1. N Tuchinda, CA Schuh (2025). [A grain boundary embrittlement genome for substitutional cubic alloys](https://pubs.aip.org/aip/apl/article/126/17/171602/3345390) (cited by 1)
1. A Giunto, Y Fei, P Nevatia et al. (2025). [Harnessing Automated SEM-EDS and Machine Learning to Unlock High-Throughput Compositional Characterization of Powder Materials](https://www.researchsquare.com/article/rs-7837297/latest) (cited by 1)
1. T Cavignac, J Schmidt, PP De Breuck et al. (2025). [AI-Driven Expansion and Application of the Alexandria Database](https://arxiv.org/abs/2512.09169) (cited by 1)
1. J Riebesell, H Yang, R Goodall et al. (2024). [janosh/pymatviz: v0. 11.0](https://ui.adsabs.harvard.edu/abs/2024zndo..13624216R/abstract) (cited by 1)
1. T Warford, FL Thiemann, GĂĄ CsĂĄnyi (2026). [Better without U: Impact of Selective Hubbard U Correction on Foundational MLIPs](https://arxiv.org/abs/2601.21056)
1. Mohammed Al-Fahdi, Riccardo Rurali, Jianjun Hu et al. (2025). [Accelerated discovery of extreme lattice thermal conductivity by crystal graph attention networks and chemical bonding](https://www.nature.com/articles/s41524-025-01871-4)
1. Omar Allam, Brook Wander, SungYeon Kim et al. (2025). [AQCat25: Unlocking spin-aware, high-fidelity machine learning potentials for heterogeneous catalysis](http://arxiv.org/abs/2510.22938)
1. Giulio Benedini, Antoine Loew, Matti Hellstrom et al. (2025). [Universal Machine Learning Potential for Systems with Reduced Dimensionality](https://arxiv.org/abs/2508.15614v1)
1. Yuan Chiang, Tobias Kreiman, Elizabeth Weaver et al. (2025). [MLIP Arena: Advancing Fairness and Transparency in Machine Learning Interatomic Potentials through an Open and Accessible Benchmark Platform](https://openreview.net/forum?id=ysKfIavYQE)
1. Orion Archer Cohen, Janosh Riebesell, Rhys Goodall et al. (2025). [TorchSim: An efficient atomistic simulation engine in PyTorch](http://iopscience.iop.org/article/10.1088/3050-287X/ae1799)
1. Alin Marin Elena, Prathami Divakar Kamath, Théo Jaffrelot Inizan et al. (2025). [Machine learned potential for high-throughput phonon calculations of metal—organic frameworks](https://www.nature.com/articles/s41524-025-01611-8)
1. Matthew K. Horton, Patrick Huck, Ruo Xi Yang et al. (2025). [Accelerated data-driven materials science with the Materials Project](https://www.nature.com/articles/s41563-025-02272-0)
1. Aaron D. Kaplan, Runze Liu, Ji Qi et al. (2025). [A Foundational Potential Energy Surface Dataset for Materials](http://arxiv.org/abs/2503.04070)
1. Matthew C. Kuner, Aaron D. Kaplan, Kristin A. Persson et al. (2025). [MP-ALOE: An r2SCAN dataset for universal machine learning interatomic potentials](http://arxiv.org/abs/2507.05559)
1. Anyang Peng, Xinzijian Liu, Ming-Yu Guo et al. (2025). [The OpenLAM Challenges: LAM Crystal Philately competition](https://doi.org/10.1088/2632-2153/add3bf)
1. Ali Ramlaoui, Martin Siron, Inel Djafar et al. (2025). [LeMat-Traj: A Scalable and Unified Dataset of Materials Trajectories for Atomistic Modeling](http://arxiv.org/abs/2508.20875)
1. Fei Shuang, Zixiong Wei, Kai Liu et al. (2025). [Universal machine learning interatomic potentials poised to supplant DFT in modeling general defects in metals and random alloys](http://arxiv.org/abs/2502.03578)
1. Yingheng Tang, Wenbin Xu, Jie Cao et al. (2025). [MatterChat: A Multi-Modal LLM for Material Science](http://arxiv.org/abs/2502.13107)
1. Liming Wu, Wenbing Huang, Rui Jiao et al. (2025). [Siamese Foundation Models for Crystal Structure Prediction](http://arxiv.org/abs/2503.10471)
1. K Yan, M Bohde, A Kryvenko (2025). [A Materials Foundation Model via Hybrid Invariant-Equivariant Architectures](https://arxiv.org/abs/2503.05771)
1. M Gibaldi, J Luo, AJ White et al. (2025). [Generalizable classification of crystal structure error types using graph attention networks](https://pubs.rsc.org/en/content/articlehtml/2025/ta/d5ta05426e)
1. Daniel W. Davies, Keith T. Butler, Adam J. Jackson et al. (2024). [SMACT: Semiconducting Materials by Analogy and Chemical Theory](https://github.com/WMD-group/SMACT)
1. Hui Zheng, Eric Sivonxay, Rasmus Christensen et al. (2024). [The ab initio non-crystalline structure database: empowering machine learning to decode diffusivity](https://www.nature.com/articles/s41524-024-01469-2)
1. Ilyes Batatia, Philipp Benner, Yuan Chiang et al. (2023). [A foundation model for atomistic materials chemistry](https://arxiv.org/abs/2401.00096v1)
1. Jack Douglas Sundberg (2022). [A New Framework for Material Informatics and Its Application Toward Electride-Halide Material Systems](https://cdr.lib.unc.edu/concern/dissertations/r494vw405)
