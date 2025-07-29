# Changelog

## [v0.17.1](https://github.com/janosh/pymatviz/compare/v0.17.0...v0.17.1)

> 29 July 2025

### 🛠 Enhancements

- Bump `matterviz` by @janosh in https://github.com/janosh/pymatviz/pull/315

### 🐛 Bug Fixes

- Fix `FileNotFoundError` `pyproject.toml` in 0.17.0 by @janosh in https://github.com/janosh/pymatviz/pull/313

## [v0.17.0](https://github.com/janosh/pymatviz/compare/v0.16.0...v0.17.0)

> 22 July 2025

### 🎉 New Features

- MatterViz widgets by @janosh in https://github.com/janosh/pymatviz/pull/311
- Sunburst of coordination numbers and environments by @janosh in https://github.com/janosh/pymatviz/pull/295

### 💥 Breaking Changes

- Add `cell_size_fn: Callable[[ModuleStats], float]` keyword to `py_pkg_treemap` by @janosh in https://github.com/janosh/pymatviz/pull/293
- Support disordered site rendering in `structure_(2|3)d_plotly()` by @janosh in https://github.com/janosh/pymatviz/pull/301
- Remove `residual_vs_actual()` by @janosh in https://github.com/janosh/pymatviz/pull/302
- `chem_env_treemap` by @janosh in https://github.com/janosh/pymatviz/pull/303
- Drop `matplotlib` dependency, refactor few remaining `matplotlib` functions to `plotly` by @janosh in https://github.com/janosh/pymatviz/pull/305

### 🛠 Enhancements

- `cluster_compositions` add `annotate_points` keyword by @janosh in https://github.com/janosh/pymatviz/pull/291
- Add element color legend to plotly 2d and 3d structure visualizations by @janosh in https://github.com/janosh/pymatviz/pull/298
- Auto-render pymatgen/ASE objects when returned from notebook cells by @janosh in https://github.com/janosh/pymatviz/pull/299
- `structure_(2|3)d_plotly` add `cell_boundary_tol: float | dict[str, float]` by @janosh in https://github.com/janosh/pymatviz/pull/300

### 🐛 Bug Fixes

- Fix plotly validators by @bmaranville in https://github.com/janosh/pymatviz/pull/310

### 🧹 House-Keeping

- Rename `structure_(2|3)d_plotly` to `structure_(2|3)d` by @janosh in https://github.com/janosh/pymatviz/pull/304
- Update `pytest-split` durations by @janosh in https://github.com/janosh/pymatviz/pull/306

### New Contributors

- @bmaranville made their first contribution in https://github.com/janosh/pymatviz/pull/310

## [v0.16.0](https://github.com/janosh/pymatviz/compare/v0.15.1...v0.16.0)

> 2 May 2025

- Add `py-pkg-treemap()` [`#290`](https://github.com/janosh/pymatviz/pull/290)
- Add `color_scale` param to allow for linear, logarithmic, arcsinh scaling of property values in `cluster_compositions` [`#288`](https://github.com/janosh/pymatviz/pull/288)
- refactor cluster_compositions to take DataFrame as 1st arg, not composition list [`#287`](https://github.com/janosh/pymatviz/pull/287)
- Test example scripts with `uv run` in CI [`#286`](https://github.com/janosh/pymatviz/pull/286)
- New `cluster` module with functions for embedding, projecting and scattering compositions [`#285`](https://github.com/janosh/pymatviz/pull/285)
- spacegroup_sunburst and chem_sys_sunburst: add keywords max_slices and max_slices_mode: other | none [`#284`](https://github.com/janosh/pymatviz/pull/284)
- Multi-trace `plotly` powerups [`#283`](https://github.com/janosh/pymatviz/pull/283)
- `brillouin_zone_3d` now supports multiple structures/atoms in customizable grid layouts [`#282`](https://github.com/janosh/pymatviz/pull/282)
- Add `max_cells: int | None` keyword to `chem_sys_treemap` [`#281`](https://github.com/janosh/pymatviz/pull/281)
- `structure_(2|3)d_plotly` enable gradient-colored bonds [`#280`](https://github.com/janosh/pymatviz/pull/280)
- Auto font color for high contrast of element symbols in `ptable_heatmap_splits_plotly` [`#279`](https://github.com/janosh/pymatviz/pull/279)
- Update `luminance` calculation to use WCAG 2.0 standard coefficients [`#278`](https://github.com/janosh/pymatviz/pull/278)
- Fix `structure_2d` `rotation` keyword not applying to unit cell [`#276`](https://github.com/janosh/pymatviz/pull/276)
- `spglib` to `moyo` [`#275`](https://github.com/janosh/pymatviz/pull/275)
- fix bond drawing in structure_3d with better image atom handling [`#274`](https://github.com/janosh/pymatviz/pull/274)
- Use SI suffix number format in ptable colorbar tick labels [`#273`](https://github.com/janosh/pymatviz/pull/273)
- fix error in docstring [`#271`](https://github.com/janosh/pymatviz/pull/271)
- Remove `matplotlib`-based periodic table plotting functions [`#270`](https://github.com/janosh/pymatviz/pull/270)
- Ward metallic glasses train/val/test splits [`#269`](https://github.com/janosh/pymatviz/pull/269)
- Codecov [`#268`](https://github.com/janosh/pymatviz/pull/268)
- Support per-split colorbars and colorscales in `ptable_heatmap_splits_plotly()` [`#267`](https://github.com/janosh/pymatviz/pull/267)
- add `enhance_parity_plot()` powerup [`#266`](https://github.com/janosh/pymatviz/pull/266)
- add treemap.py with new chem_sys_treemap plot function [`#265`](https://github.com/janosh/pymatviz/pull/265)
- add chem_sys_sunburst() to pymatviz/sunburst.py to visualize chemical system distributions [`#264`](https://github.com/janosh/pymatviz/pull/264)
- test metallic glass feature engineering and model evaluation [`f0c5174`](https://github.com/janosh/pymatviz/commit/f0c517419cfb6bd9fafc8d09387474b119cb875e)
- better colorbar tick formatting and spacing in density_scatter_plotly [`d1d51b4`](https://github.com/janosh/pymatviz/commit/d1d51b47eeba7a3cc200d27f5b363447807ea720)
- RDF plotting hide legend for single structure [`dc8a1a2`](https://github.com/janosh/pymatviz/commit/dc8a1a2d53050a374eb819706d9864f409585960)
- fix confusion_matrix() mismatch of false-positive/negative counts and heatmap color [`ae5b629`](https://github.com/janosh/pymatviz/commit/ae5b629b74d0273d01a1851cec86f5ef1e0641e3)
- `contrast_ratio()` in `pymatviz/utils/plotting.py` to calculate color contrast according to WCAG 2.0 [`b7c30cf`](https://github.com/janosh/pymatviz/commit/b7c30cf4699f97c6b67c6e6a01e86c0cccd8286c)
- Add script to fetch and update papers citing pymatviz (#277) [`6a2b831`](https://github.com/janosh/pymatviz/commit/6a2b8317757858f9b599e6a2362c8e365339dfac)
- add examples/mlip_phonons.py [`608a12f`](https://github.com/janosh/pymatviz/commit/608a12f0d06f4623de84973a102057166d2f00ef)
- clean up examples/diatomics/calc_mlip_diatomic_curves.py and support multiple ML models [`d6fe3d9`](https://github.com/janosh/pymatviz/commit/d6fe3d926606a1f87d6f7314c0b6ffb6b8d5bd85)
- add examples/compare_elastic_constants.py testing MACE vs MP PBE [`2861298`](https://github.com/janosh/pymatviz/commit/2861298ef3176f87885a1329fdee760e4ce300d0)
- Refactor ROC and PR curve no-skill line plotting [`6d68cc1`](https://github.com/janosh/pymatviz/commit/6d68cc174e7aea49863d29ac7ac997b48d5d6136)
- `readme.md` add links to source and example code for each plot function [`65ccb7c`](https://github.com/janosh/pymatviz/commit/65ccb7cb5bda434321f939345b83bb4fbc4d81bd)
- add assets/scripts/key_enum_table.py to visualize Key enum attributes with Plotly [`0598bb5`](https://github.com/janosh/pymatviz/commit/0598bb50b0a1ebb682a1b17f7e380877792c7072)
- delete unused enums Model + ElemColorMode [`ee296bc`](https://github.com/janosh/pymatviz/commit/ee296bc7a588a481c39204db7db1f156cc749fd9)
- fix ptable_heatmap_splits_plotly() incorrectly handling hide_f_block keyword resulting in missing tiles for Rf 104 through Og 118 [`b9ccbf0`](https://github.com/janosh/pymatviz/commit/b9ccbf0ee3017b066e16640e6749845aff7979c0)
- force LabelEnums to have labels, gives .label type str causing less mypy headache [`a6825ff`](https://github.com/janosh/pymatviz/commit/a6825ff780416aec660510c295a1b958f68010eb)

## [v0.15.1](https://github.com/janosh/pymatviz/compare/v0.15.0...v0.15.1)

> 28 January 2025

- Remove hard-coded `gridsize` in `density_hexbin` [`#263`](https://github.com/janosh/pymatviz/pull/263)
- structure_2d and structure_3d now support ase.Atoms and sequences of them on top of pymatgen.Structure [`#262`](https://github.com/janosh/pymatviz/pull/262)
- add molecular dynamics attributes to Key enum [`4b55a40`](https://github.com/janosh/pymatviz/commit/4b55a400e639b696c25dc752a0a70c0e472740df)
- better Key.unit formatting: replace unicode sup/superscripts with &lt;sup&gt;/&lt;sub&gt;+ASCII [`eb12217`](https://github.com/janosh/pymatviz/commit/eb122173df12cfcbcff1070a1a6177a581de248b)
- bump ruff to 0.9 and auto-fix [`6989ba5`](https://github.com/janosh/pymatviz/commit/6989ba5496dcf5f699f3a64ef38dd33f65f3ac28)
- calculate MACE-MPA-0 diatomic curves [`4dc5f1e`](https://github.com/janosh/pymatviz/commit/4dc5f1ee095d398089443a543519a386532d49c5)

## [v0.15.0](https://github.com/janosh/pymatviz/compare/v0.14.0...v0.15.0)

> 21 December 2024

- Multi-line `ptable_scatter_plotly` [`#260`](https://github.com/janosh/pymatviz/pull/260)
- Hetero-nuclear diatomics example with MACE [`#259`](https://github.com/janosh/pymatviz/pull/259)
- Add `ptable_scatter_plotly` [`#258`](https://github.com/janosh/pymatviz/pull/258)
- Remove `cumulative.py` and associated tests + assets [`#257`](https://github.com/janosh/pymatviz/pull/257)
- Support `phonopy` `TotalDos` and `BandStructure` in `phonon_dos` and `phonon_bands` plots [`#256`](https://github.com/janosh/pymatviz/pull/256)
- Add `brillouin_zone_3d` plot function [`#251`](https://github.com/janosh/pymatviz/pull/251)
- Add element color schemes `alloys` [`#255`](https://github.com/janosh/pymatviz/pull/255)
- delete pymatviz/classify/curves_matplotlib.py [`#254`](https://github.com/janosh/pymatviz/pull/254)
- `plotly` ROC and precision-recall curves [`#253`](https://github.com/janosh/pymatviz/pull/253)
- Add `pymatviz.classify.confusion_matrix` [`#252`](https://github.com/janosh/pymatviz/pull/252)
- Fix `phonon_bands` for band structures with different paths in k-space [`#250`](https://github.com/janosh/pymatviz/pull/250)
- keys.yml add more related to electronic, mechanical, thermal, and magnetic properties [`f377d95`](https://github.com/janosh/pymatviz/commit/f377d95aea1127f39108aa8a796f9fac13857d59)
- classify/curves_plotly.py show threshold on hover, add_add_no_skill_line helper [`77e765a`](https://github.com/janosh/pymatviz/commit/77e765a1b6fc44006e2b684e2a4a76f86f2b4c52)
- breaking: rename Key.cse to computed_structure_entry [`ab691bb`](https://github.com/janosh/pymatviz/commit/ab691bbfa9f42151a42bcc3a6b2e0a261820b8b6)
- test_brillouin_zone_3d_trace_counts [`44353aa`](https://github.com/janosh/pymatviz/commit/44353aa5c02259ab82b1d42e1ac3ac9e02ef5a18)

## [v0.14.0](https://github.com/janosh/pymatviz/compare/v0.13.2...v0.14.0)

> 21 November 2024

- [Breaking] Split `utils` into sub-modules, move `typing` from `utils` to root (`pmv.typing`) [`#248`](https://github.com/janosh/pymatviz/pull/248)
- `phonon_bands` enable custom acoustic/optical bands [`#249`](https://github.com/janosh/pymatviz/pull/249)
- Split `make_assets` scripts by plot functions [`#247`](https://github.com/janosh/pymatviz/pull/247)
- `ptable_heatmap_splits_plotly` [`#246`](https://github.com/janosh/pymatviz/pull/246)
- Better `ptable_hists_plotly` defaults [`#244`](https://github.com/janosh/pymatviz/pull/244)
- Add `ptable_hists_plotly` [`#243`](https://github.com/janosh/pymatviz/pull/243)
- Enhance warning message for default return type change of `ptable_heatmap` [`#240`](https://github.com/janosh/pymatviz/pull/240)
- add @stylistic/eslint-plugin to fix eslint commit hook (closes https://github.com/janosh/pymatviz/issues/197) [`#197`](https://github.com/janosh/pymatviz/issues/197)
- add coordination_nums_in_structure in pymatviz/coordination/helpers.py [`bd54679`](https://github.com/janosh/pymatviz/commit/bd54679c17c6c137949a162dce25770587ebe200)
- ptable_hists_plotly add x_axis_kwargs: dict[str, Any] | None = None to tweak x ticks and allow annotations renamed to be callable [`6a6faad`](https://github.com/janosh/pymatviz/commit/6a6faadeb2b5d5a8be9d8e6f4383f122c7e5373d)
- sankey_from_2_df_cols add kwarg annotate_columns: bool | dict = True [`eed306a`](https://github.com/janosh/pymatviz/commit/eed306a621e32584064f62be37aa7a1cd2db50a4)
- new/renamed Key attributes n_structs, n_materials, n_molecules, n_samples, n_configs [`786a666`](https://github.com/janosh/pymatviz/commit/786a6667585b40b583e816bae7b857294418a735)

## [v0.13.2](https://github.com/janosh/pymatviz/compare/v0.13.1...v0.13.2)

> 3 November 2024

- Split `coordination.py` module into `plotly.py` and `helpers.py` [`#239`](https://github.com/janosh/pymatviz/pull/239)
- Fix `ptable_heatmap_ratio` following change of default `count_elements` `fill_value` to 0 [`#226`](https://github.com/janosh/pymatviz/pull/226)
- Speedup import and add regression check for import time [`#238`](https://github.com/janosh/pymatviz/pull/238)
- address coordination_hist TODO 'get the right y_max when bar_mode="stack"' [`f73ba99`](https://github.com/janosh/pymatviz/commit/f73ba99e691aa9c861d34641b87ac0113c8b54f2)
- new Key enum attributes for DFT/experimental settings and paper/code metadata [`8426bc2`](https://github.com/janosh/pymatviz/commit/8426bc2bd45f9ea5ff7a521ee3da3d7f8039293a)
- dynamic element_pair_rdfs cutoff, defaults to 2x max_cell_len, allow negative values interpreted as scaling factor to max_cell_len (instead of absolute length in Angstrom) [`d6b1b01`](https://github.com/janosh/pymatviz/commit/d6b1b0153e90cff2c6a32bbb2db8ac76aae30ae0)

## [v0.13.1](https://github.com/janosh/pymatviz/compare/v0.13.0...v0.13.1)

> 18 October 2024

- fix ImportError: cannot import pymatgen.symmetry.analyzer.SymmetryUndeterminedError [`298c49a`](https://github.com/janosh/pymatviz/commit/298c49a60a308ced2da54bc2d17ce997694c4ed6)

## [v0.13.0](https://github.com/janosh/pymatviz/compare/v0.12.0...v0.13.0)

> 17 October 2024

- Fix `/api` docs by migrating to `remark`/`rehype` [`#237`](https://github.com/janosh/pymatviz/pull/237)
- Add `coordination_vs_cutoff_line` plot function [`#236`](https://github.com/janosh/pymatviz/pull/236)
- Reduce duplicate in dependency declare [`#231`](https://github.com/janosh/pymatviz/pull/231)
- Coordination number histograms [`#235`](https://github.com/janosh/pymatviz/pull/235)
- Add `full_rdf` [`#234`](https://github.com/janosh/pymatviz/pull/234)
- Experimental: add `show_bonds: bool | NearNeighbors = False` to `structure_(2|3)d_plotly` [`#233`](https://github.com/janosh/pymatviz/pull/233)
- drop correlation.py module [`#229`](https://github.com/janosh/pymatviz/pull/229)
- Custom site hover texts in `pmv.structure_(2|3)d_plotly` [`#228`](https://github.com/janosh/pymatviz/pull/228)
- Rainclouds [`#227`](https://github.com/janosh/pymatviz/pull/227)
- replace "sankey-crystal-sys-to-spg-symbol" with "sankey-spglib-vs-aflow-spacegroups" [`3d818a1`](https://github.com/janosh/pymatviz/commit/3d818a15c08d0396b99a12a469ce772f2b716f6d)
- more Key enum attributes for various errors and number of steps properties [`701a445`](https://github.com/janosh/pymatviz/commit/701a44583e81673e34b077e2429582e91dcbd6da)

## [v0.12.0](https://github.com/janosh/pymatviz/compare/v0.11.0...v0.12.0)

> 7 October 2024

- use `pytest-split` in GitHub Action [`#224`](https://github.com/janosh/pymatviz/pull/224)
- Vertically/horizontally stacked XRD plots [`#223`](https://github.com/janosh/pymatviz/pull/223)
- add scaling_factor to pymatviz/ptable/plotly.py [`#210`](https://github.com/janosh/pymatviz/pull/210)
- Support plotting site vectors like forces/magmoms in `structure_(2|3)d_plotly` [`#220`](https://github.com/janosh/pymatviz/pull/220)
- Add `add_annotation` functionality for ptable plotters [`#200`](https://github.com/janosh/pymatviz/pull/200)
- Render spheres with hover tooltip on unit cell corners in `pmv.structure_(2|3)d_plotly` [`#219`](https://github.com/janosh/pymatviz/pull/219)
- `structure_(2|3)d_plotly` allow overriding subplot title's y position and anchor [`#218`](https://github.com/janosh/pymatviz/pull/218)
- pin kaleido==0.2.1 [`#217`](https://github.com/janosh/pymatviz/pull/217)
- Add `structure_3d` [`#214`](https://github.com/janosh/pymatviz/pull/214)
- Clean up var names in unit tests, avoid MP API access in GitHub workflow [`#207`](https://github.com/janosh/pymatviz/pull/207)
- Add `structure_2d` to `pymatviz/structure_viz.py` [`#213`](https://github.com/janosh/pymatviz/pull/213)
- Fix `xrd_pattern` not allowing `annotate_peaks=0` to disable peak annotation [`#212`](https://github.com/janosh/pymatviz/pull/212)
- Fix `calculate_rdf` not accounting for periodic boundaries [`#211`](https://github.com/janosh/pymatviz/pull/211)
- Support `list`/`dict` of structures in `element_pair_rdfs` [`#206`](https://github.com/janosh/pymatviz/pull/206)
- Add kwarg `use_tooltips: bool = True` to `df_to_html` (prev `df_to_html_table`) [`#205`](https://github.com/janosh/pymatviz/pull/205)
- Exclude `tests` from source distribution, and drop python2 tag for wheel [`#202`](https://github.com/janosh/pymatviz/pull/202)
- `element_pair_rdfs` plots radial distribution functions (RDFs) for element pairs in a structure [`#203`](https://github.com/janosh/pymatviz/pull/203)
- Add `IS_IPYTHON` global [`#198`](https://github.com/janosh/pymatviz/pull/198)
- ruff fixes [`#196`](https://github.com/janosh/pymatviz/pull/196)
- breaking: rename &lt;xyz&gt;_kwds -&gt; &lt;xyz&gt;_kwargs for consistency [`24261ca`](https://github.com/janosh/pymatviz/commit/24261caea8af69ab9698308ce6034bd8ca67b3c1)

## [v0.11.0](https://github.com/janosh/pymatviz/compare/v0.10.1...v0.11.0)

> 1 September 2024

- Bump min supported Python to 3.10 [`#195`](https://github.com/janosh/pymatviz/pull/195)
- fix ptable_heatmap return type plt.axes-&gt;plt.Axes [`1de350a`](https://github.com/janosh/pymatviz/commit/1de350a146b53ae03f343ff9a78454ba7c976186)

## [v0.10.1](https://github.com/janosh/pymatviz/compare/v0.10.0...v0.10.1)

> 18 August 2024

- Self-import refactor [`#194`](https://github.com/janosh/pymatviz/pull/194)
- Fix `svgo` workflow for ptable scatter plots [`#187`](https://github.com/janosh/pymatviz/pull/187)
- `density_scatter_plotly` add kwarg `facet_col: str | None = None` [`#193`](https://github.com/janosh/pymatviz/pull/193)
- `bin_df_cols` leave input df unchanged [`#192`](https://github.com/janosh/pymatviz/pull/192)
- Re-export all submodules/subpackages from `pymatviz.__init__.py` [`#191`](https://github.com/janosh/pymatviz/pull/191)
- fix missing jinja2 dep at import time [`fb6c9df`](https://github.com/janosh/pymatviz/commit/fb6c9df42e4d40cccca305ca406f8a9975a27a78)

## [v0.10.0](https://github.com/janosh/pymatviz/compare/v0.9.3...v0.10.0)

> 31 July 2024

- `density_scatter_plotly` QoL tweaks [`#190`](https://github.com/janosh/pymatviz/pull/190)
- Breaking: drop `plot_` prefix from multiple functions [`#189`](https://github.com/janosh/pymatviz/pull/189)
- Add protostructure terminology to enum [`#185`](https://github.com/janosh/pymatviz/pull/185)
- Update the function names for wren utils given breaking changes in Aviary [`#182`](https://github.com/janosh/pymatviz/pull/182)
- [Breaking] Refactor ptable heatmap plotter [`#157`](https://github.com/janosh/pymatviz/pull/157)

## [v0.9.3](https://github.com/janosh/pymatviz/compare/v0.9.2...v0.9.3)

> 18 July 2024

- Fix ptable scatter examples in homepage [`#180`](https://github.com/janosh/pymatviz/pull/180)
- Add `pymatviz.io.df_to_svg` [`#179`](https://github.com/janosh/pymatviz/pull/179)
- Better default `ptable_heatmap_plotly` tooltips [`#178`](https://github.com/janosh/pymatviz/pull/178)
- remove skip tag for tests [`#177`](https://github.com/janosh/pymatviz/pull/177)
- Moving enums may have broken end users pickle's, reduce to str when pickling to be more backwards compatible going forward. [`#176`](https://github.com/janosh/pymatviz/pull/176)
- Better `density_scatter_plotly` [`#175`](https://github.com/janosh/pymatviz/pull/175)
- new ML model, metrics and computational details related enum keys [`b6cdca3`](https://github.com/janosh/pymatviz/commit/b6cdca34a3643aa9dc04070bb92470bb20e7ad3e)

## [v0.9.2](https://github.com/janosh/pymatviz/compare/v0.9.1...v0.9.2)

> 7 July 2024

- Fix `ptable_heatmap_plotly` for `log=True` [`#174`](https://github.com/janosh/pymatviz/pull/174)
- Fix missing keys `Te` + `Nd` in `ELEM_COLORS_VESTA` and support it in `plot_structure_2d` [`#173`](https://github.com/janosh/pymatviz/pull/173)
- Fix `log_density` in `density_scatter_plotly` [`#172`](https://github.com/janosh/pymatviz/pull/172)

## [v0.9.1](https://github.com/janosh/pymatviz/compare/v0.9.0...v0.9.1)

> 4 July 2024

- Split `powerups` module by `backend`: `matplotlib`/`plotly`/`both` [`#171`](https://github.com/janosh/pymatviz/pull/171)
- Fix `count_elements` for series of `Composition` [`#170`](https://github.com/janosh/pymatviz/pull/170)
- Fix and test `ptable_heatmap` text color logic [`#169`](https://github.com/janosh/pymatviz/pull/169)
- `plot_xrd_pattern` accept `DiffractionPattern | Structure` as input [`#168`](https://github.com/janosh/pymatviz/pull/168)
- Add `plot_xrd_pattern()` for creating interactive XRD patterns with plotly [`#167`](https://github.com/janosh/pymatviz/pull/167)
- Fix `density_scatter_plotly` metric annotation [`#166`](https://github.com/janosh/pymatviz/pull/166)
- Add `toggle_log_linear_y_axis` powerup [`#165`](https://github.com/janosh/pymatviz/pull/165)
- Fix bad NPY002 migration [`#163`](https://github.com/janosh/pymatviz/pull/163)
- refactor to explicit ax passing instead of relying on plt.gca() in example scripts [`7912ec0`](https://github.com/janosh/pymatviz/commit/7912ec004cf2201a372cec896998448beac594ba)
- add_ecdf_line improve adapting trace_defaults from target_trace [`4b451a3`](https://github.com/janosh/pymatviz/commit/4b451a38d77c34565004487eccadf7cfb0cc9140)

## [v0.9.0](https://github.com/janosh/pymatviz/compare/v0.8.3...v0.9.0)

> 21 June 2024

- Fix `ruff` `NPY002` [`#162`](https://github.com/janosh/pymatviz/pull/162)
- `density_scatter_plotly()` [`#161`](https://github.com/janosh/pymatviz/pull/161)
- Add `pymatviz.histogram.plot_histogram` [`#159`](https://github.com/janosh/pymatviz/pull/159)
- Fix `ptable_heatmap_ratio` legend [`#158`](https://github.com/janosh/pymatviz/pull/158)
- `ptable_scatters` allow 3rd data dimension for colormap [`#155`](https://github.com/janosh/pymatviz/pull/155)
- Support passing sequence of structures to `plot_structure_2d()` to be plotted in grid [`#156`](https://github.com/janosh/pymatviz/pull/156)
- `density_scatter_plotly()` (#161) [`#160`](https://github.com/janosh/pymatviz/issues/160)
- when passed a series, plot_histogram now use series name as x-axis title [`e9697fc`](https://github.com/janosh/pymatviz/commit/e9697fcbca24c6b10d11069ac6a969b5d9507fe1)
- import std lib StrEnum from enum if sys.version_info &gt;= (3, 11) [`510452a`](https://github.com/janosh/pymatviz/commit/510452aeb9a57e33283a8bedd530ddf77a6fe4ce)
- add plot_structure_2d() keyword subplot_title: Callable[[Structure, str | int], str] | None = None [`523101b`](https://github.com/janosh/pymatviz/commit/523101bc6af2e0b9168a1775cd9c95dd09c3ec4b)

## [v0.8.3](https://github.com/janosh/pymatviz/compare/v0.8.2...v0.8.3)

> 30 May 2024

- Fix `PTableProjector.hide_f_block` property [`#154`](https://github.com/janosh/pymatviz/pull/154)
- Handle missing value (NaN) and infinity for ptable data [`#152`](https://github.com/janosh/pymatviz/pull/152)
- [Enhancement/Breaking] Refactor `ptable_hists` [`#149`](https://github.com/janosh/pymatviz/pull/149)
- Add keyword `log: bool = False` to `spacegroup_hist` to log scale y-axis [`#148`](https://github.com/janosh/pymatviz/pull/148)
- MatPES EDA script [`#147`](https://github.com/janosh/pymatviz/pull/147)
- Fix `ptable_heatmap_splits` `TypeErrors` [`#146`](https://github.com/janosh/pymatviz/pull/146)
- Add `mlff_phonons.ipynb` example notebook [`#144`](https://github.com/janosh/pymatviz/pull/144)
- split examples/_generate_assets.py by pymatviz module whose plot functions are made assets for [`ef3f066`](https://github.com/janosh/pymatviz/commit/ef3f066a4d80589a724d552903a54da1edeb6164)
- show ptable-heatmap-splits-3 in readme [`1be63ea`](https://github.com/janosh/pymatviz/commit/1be63eadd824fb726d507ff13bbfaf73538e9c98)
- fix docs build: ParseError: Unexpected character '“' [`fbb2c8f`](https://github.com/janosh/pymatviz/commit/fbb2c8f67d483b48d6adfd768f80717c98e74061)

## [v0.8.2](https://github.com/janosh/pymatviz/compare/v0.8.1...v0.8.2)

> 11 May 2024

- Add `pymatviz/enums.py` for SSOT on dataframe column and dict key names [`#143`](https://github.com/janosh/pymatviz/pull/143)
- Add keyword `hide_f_block: bool = None` (La and Ac series) to `ptable` plotters [`#140`](https://github.com/janosh/pymatviz/pull/140)
- Remove text background and fix z-order in `structure_viz` [`#139`](https://github.com/janosh/pymatviz/pull/139)
- Refactor `ptable` plotters and add `ptable_heatmap` with diagonally-split tiles [`#131`](https://github.com/janosh/pymatviz/pull/131)
- Migrate to flat `eslint` config file [`#137`](https://github.com/janosh/pymatviz/pull/137)
- Copy color options for element types from `ptable_plots` to `ptable_hists` [`#129`](https://github.com/janosh/pymatviz/pull/129)
- MACE-MP pair repulsion curves [`#127`](https://github.com/janosh/pymatviz/pull/127)
- Add `validate_fig` decorator utility [`#126`](https://github.com/janosh/pymatviz/pull/126)
- `add_best_fit_line()` power-up [`#125`](https://github.com/janosh/pymatviz/pull/125)
- `plot_phonon_bands()` add kwargs `branch_mode: "union" | "intersection" = "union"` and `branches: Sequence[str] = ()` [`#124`](https://github.com/janosh/pymatviz/pull/124)
- breaking: rename fmt_spec keyword to fmt in si_fmt() for code base consistency [`4d3091e`](https://github.com/janosh/pymatviz/commit/4d3091edb209621c2dcc0d299f34a1b039fea8de)
- add decimal_threshold: float = 0.01 to si_fmt() [`6ecf68a`](https://github.com/janosh/pymatviz/commit/6ecf68a5f449d8c6b93a16c553d6233433612948)
- fix docs build error and address bullet point 3 in #138 [`492cd53`](https://github.com/janosh/pymatviz/commit/492cd5301f9b241291a49f24a4d8adbbd50ae5c6)
- update bibtex citation author list [`aa132d9`](https://github.com/janosh/pymatviz/commit/aa132d92f28ec76a22b7bfe180dc97317592dd37)

## [v0.8.1](https://github.com/janosh/pymatviz/compare/v0.8.0...v0.8.1)

> 11 February 2024

- Breaking: rename custom plotly template `pymatviz_(black-&gt;dark)` [`#123`](https://github.com/janosh/pymatviz/pull/123)
- Add ptable_scatter [`#122`](https://github.com/janosh/pymatviz/pull/122)
- Minor format tweaks [`#120`](https://github.com/janosh/pymatviz/pull/120)
- Add `pytest` fixtures `df_(float|mixed)` to replace deleted `pd._testing.make(Mixed)DataFrame()` [`#121`](https://github.com/janosh/pymatviz/pull/121)
- `add_ecdf_line()` utility for plotting empirical cumulative distribution functions [`#117`](https://github.com/janosh/pymatviz/pull/117)
- Spacegroup hist plotly [`#116`](https://github.com/janosh/pymatviz/pull/116)
- Add `plot_phonon_bands_and_dos()` [`#115`](https://github.com/janosh/pymatviz/pull/115)
- Add `show_values: bool = True` to `ptable_heatmap_plotly()` and `last_peak_anno: str` to `plot_phonon_dos()` [`#114`](https://github.com/janosh/pymatviz/pull/114)
- Add `plot_phonon_dos()` for interactive plotly DOS plots [`#113`](https://github.com/janosh/pymatviz/pull/113)
- _generate_assets.py add code for assets/(phonon-bands-and-dos-dft|phonon-bands-dft|phonon-dos-dft).svg [`93f72dc`](https://github.com/janosh/pymatviz/commit/93f72dcf46ab67a7713541b903f56d690e7791da)
- ruff enable PD901+PLW2901 and fix violations [`36f4771`](https://github.com/janosh/pymatviz/commit/36f477103cfc82109bbb175a052f0fc22109f0d3)
- breaking: absorb keywords sort and density_bins into hist_density_kwargs [`64da6b2`](https://github.com/janosh/pymatviz/commit/64da6b2da3d34ecda8028adc0a874c364e68bf62)
- breaking: rename elements_hist to hist_elemental_prevalence in pymatviz/histograms.py [`ebd067a`](https://github.com/janosh/pymatviz/commit/ebd067aa8cee15a7de44676fb37e2f3e472439eb)
- gray-shade negative frequencies in phonon bands and bands+DOS plots [`882a6ca`](https://github.com/janosh/pymatviz/commit/882a6caaa06acbcadbcfc3d8e2d7fe6aa8ce630e)
- breaking: remove residual_hist() from pymatviz/histograms.py [`7780f68`](https://github.com/janosh/pymatviz/commit/7780f68d593dcdabe750a6b2b868f3bddfea7476)
- mv dataset_exploration examples [`8bfa4b8`](https://github.com/janosh/pymatviz/commit/8bfa4b8c958db1fd7f3877ad7c8d2a2864ba6936)

## [v0.8.0](https://github.com/janosh/pymatviz/compare/v0.7.3...v0.8.0)

> 15 December 2023

- Add `plot_band_structure` in new `pymatviz/bandstructure.py` module [`#112`](https://github.com/janosh/pymatviz/pull/112)
- Add `hist_kwds` arg to `ptable_hists` to customize histograms [`#111`](https://github.com/janosh/pymatviz/pull/111)
- Define custom `pymatviz` plotly templates [`#110`](https://github.com/janosh/pymatviz/pull/110)
- Support matplotlib `Axes` and `Figure` in `add_identity_line` [`#109`](https://github.com/janosh/pymatviz/pull/109)
- Tweaks [`#108`](https://github.com/janosh/pymatviz/pull/108)
- Add function `ptable_hists` [`#100`](https://github.com/janosh/pymatviz/pull/100)
- `ptable_heatmap_plotly` support 1s, 0s and negative values with `log=True` [`#107`](https://github.com/janosh/pymatviz/pull/107)
- `ptable_heatmap` add keywords `cbar_range` and `cbar_kwargs` [`#105`](https://github.com/janosh/pymatviz/pull/105)
- Add class `TqdmDownload` [`#104`](https://github.com/janosh/pymatviz/pull/104)
- Breaking: rename `get_crystal_sys` to `crystal_sys_from_spg_num` [`#103`](https://github.com/janosh/pymatviz/pull/103)
- Support semi-log + log-log plots in `add_identity_line` [`#102`](https://github.com/janosh/pymatviz/pull/102)
- `plot_structure_2d` add special `site_labels: "symbol" | "species"` [`#101`](https://github.com/janosh/pymatviz/pull/101)
- ptable_heatmap_ratio allow disabling not_in_numerator, not_in_denominator, not_in_numerator [`c4bc03d`](https://github.com/janosh/pymatviz/commit/c4bc03d3b8a6202ab44e2c1bd41bca703608cde4)
- fix add_identity_line for log-scaled matplotlib Axes: TypeError: 'slope' cannot be used with non-linear scales [`ead0ce9`](https://github.com/janosh/pymatviz/commit/ead0ce916f2a30b602a0310b14d7edd99cde52aa)
- add and test si_fmt_int in pymatviz/utils.py [`fc40cd7`](https://github.com/janosh/pymatviz/commit/fc40cd7ec93af1a81cd9081b68f96f43be9a8fea)

## [v0.7.3](https://github.com/janosh/pymatviz/compare/v0.7.2...v0.7.3)

> 4 November 2023

- Add `styled_html_tag()` in `utils.py` [`#99`](https://github.com/janosh/pymatviz/pull/99)
- Add `si_fmt()` for formatting large numbers in human-readable format [`#98`](https://github.com/janosh/pymatviz/pull/98)

## [v0.7.2](https://github.com/janosh/pymatviz/compare/v0.7.1...v0.7.2)

> 30 October 2023

- Fix and rename `df_to_(svelte-&gt;html)_table` [`#97`](https://github.com/janosh/pymatviz/pull/97)
- Add keyword `default_styles: bool = True` to `df_to_pdf` [`#96`](https://github.com/janosh/pymatviz/pull/96)

## [v0.7.1](https://github.com/janosh/pymatviz/compare/v0.7.0...v0.7.1)

> 22 October 2023

- Periodic table UX improvements [`#95`](https://github.com/janosh/pymatviz/pull/95)
- `annotate_bars` add keyword `adjust_test_pos: bool = False` [`#94`](https://github.com/janosh/pymatviz/pull/94)
- Add `df_to_svelte_table` [`#93`](https://github.com/janosh/pymatviz/pull/93)
- fix invalid count_mode ValueError err msg [`ae77997`](https://github.com/janosh/pymatviz/commit/ae779970e210e382006b7a730175ac025b026285)

## [v0.7.0](https://github.com/janosh/pymatviz/compare/v0.6.3...v0.7.0)

> 9 October 2023

- Bump minimum Python version to 3.9 [`#92`](https://github.com/janosh/pymatviz/pull/92)
- Split `pymatviz/io.py` out from `pymatviz/utils.py` and add `df_to_pdf()` export function [`#91`](https://github.com/janosh/pymatviz/pull/91)
- Add KDE support to `bin_df_cols` utility function [`#90`](https://github.com/janosh/pymatviz/pull/90)
- Add `patch_dict()` utility [`#88`](https://github.com/janosh/pymatviz/pull/88)
- Breaking: rename `ptable_heatmap` and `annotate_metrics` float precision kwargs to `fmt` [`#87`](https://github.com/janosh/pymatviz/pull/87)
- Rename `ptable_heatmap(_plotly)` 1st arg: `elem_values-&gt;values` [`#86`](https://github.com/janosh/pymatviz/pull/86)

## [v0.6.3](https://github.com/janosh/pymatviz/compare/v0.6.2...v0.6.3)

> 24 July 2023

- Fix "Loading [MathJax]/extensions/MathMenu.js" in Plotly figures exported to PDF [`#83`](https://github.com/janosh/pymatviz/pull/83)
- Tiny doc update in ptable.py [`#82`](https://github.com/janosh/pymatviz/pull/82)
- Better type errors [`#80`](https://github.com/janosh/pymatviz/pull/80)
- `ruff` enable more rule sets [`#79`](https://github.com/janosh/pymatviz/pull/79)
- Disable `save_fig()` in CI [`#78`](https://github.com/janosh/pymatviz/pull/78)
- adhere to PEP 484 (no implicit optional) [`8e50218`](https://github.com/janosh/pymatviz/commit/8e5021876b65f13d10b34e65fc9738b16489bee4)
- fix ruff TCH002,TCH003 [`d60276b`](https://github.com/janosh/pymatviz/commit/d60276bfc55d69d138c8784e1b29a0e658b32e5e)
- add ptable_heatmap_plotly kwarg label_map: dict[str, str] | False | None = None [`ef40171`](https://github.com/janosh/pymatviz/commit/ef401718a2903bbc1f64364e686d15b9ae614988)
- migrate site to eslint-plugin-svelte [`91d7909`](https://github.com/janosh/pymatviz/commit/91d7909362dab8739d0b84579e6134766b205bce)

## [v0.6.2](https://github.com/janosh/pymatviz/compare/v0.6.1...v0.6.2)

> 29 April 2023

- Per-module doc pages [`#77`](https://github.com/janosh/pymatviz/pull/77)
- Refactor `make_docs.py` [`#76`](https://github.com/janosh/pymatviz/pull/76)
- DRY workflows [`#74`](https://github.com/janosh/pymatviz/pull/74)
- More flexible `annotate_metrics` util [`#73`](https://github.com/janosh/pymatviz/pull/73)

## [v0.6.1](https://github.com/janosh/pymatviz/compare/v0.6.0...v0.6.1)

> 21 March 2023

- Add kwarg `axis: bool | str = "off"` to `plot_structure_2d()` [`#72`](https://github.com/janosh/pymatviz/pull/72)
- Add `ptable_heatmap` `cbar_precision` kwarg [`#70`](https://github.com/janosh/pymatviz/pull/70)
- add changelog.md via auto-changelog [`05da617`](https://github.com/janosh/pymatviz/commit/05da61795d3ecb67026f964be79a36cf0737b760)
- add half-baked /plots and /notebook pages [`ed171ec`](https://github.com/janosh/pymatviz/commit/ed171ec947bfcdbbd5bfda85e2ba6b48e0e35e39)
- add svelte-zoo PrevNext to notebooks pages [`05368c0`](https://github.com/janosh/pymatviz/commit/05368c034f6c2aff572011b93c5b4c722ee7559b)
- add new option 'occurrence' for CountMode = element_composition|fractional_composition|reduced_composition [`bf1604a`](https://github.com/janosh/pymatviz/commit/bf1604a27b2dc80d657ae36a0070d83cba4e2e86)
- refactor ptable_heatmap()'s tick_fmt() and add test for cbar_precision kwarg [`3427e1f`](https://github.com/janosh/pymatviz/commit/3427e1fce2ccd13aacf7b2043d9c10c6375027af)
- plot_structure_2d() in site /api docs [`fcf75de`](https://github.com/janosh/pymatviz/commit/fcf75de255cb3a3fb1555b2b1f7595e0be66043d)

## [v0.6.0](https://github.com/janosh/pymatviz/compare/v0.5.3...v0.6.0)

> 21 February 2023

- Pyproject [`#69`](https://github.com/janosh/pymatviz/pull/69)

## [v0.5.3](https://github.com/janosh/pymatviz/compare/v0.5.2...v0.5.3)

> 20 February 2023

- Pyproject [`#69`](https://github.com/janosh/pymatviz/pull/69)
- Add Ruff pre-commit hook [`#68`](https://github.com/janosh/pymatviz/pull/68)
- scatter_density() use x, y args as axis labels if strings [`0f2386a`](https://github.com/janosh/pymatviz/commit/0f2386a826855f6a961e1ad356bcc567ba2c2c88)
- fix util save_and_compress_svg() and update plot_structure_2d() assets [`e0020aa`](https://github.com/janosh/pymatviz/commit/e0020aa25f45ad7dbcad4c8c2cfc8eb5542266d9)
- use redirect in layout.ts instead of ugly DOM href surgery to forward readme links to GH repo [`7da3c0c`](https://github.com/janosh/pymatviz/commit/7da3c0c51ea43871a1a8ee6d9886506d16a5f601)
- rename add_mae_r2_box() to annotate_mae_r2() [`c550332`](https://github.com/janosh/pymatviz/commit/c550332700ed630ce1aa375fa5c2ba45022ccb10)

## [v0.5.2](https://github.com/janosh/pymatviz/compare/v0.5.1...v0.5.2)

> 13 January 2023

- pre-commit autoupdate [`#65`](https://github.com/janosh/pymatviz/pull/65)
- Deploy demo site to GitHub pages [`#64`](https://github.com/janosh/pymatviz/pull/64)
- Configure `devcontainer` for running notebooks in Codespace [`#63`](https://github.com/janosh/pymatviz/pull/63)
- Customizable parity stats [`#61`](https://github.com/janosh/pymatviz/pull/61)
- Support dataframes in `true_pred_hist()` [`#60`](https://github.com/janosh/pymatviz/pull/60)
- Support dataframes in relevance and uncertainty plots [`#59`](https://github.com/janosh/pymatviz/pull/59)
- Allow passing in dataframes and x, y as column names in parity plots [`#58`](https://github.com/janosh/pymatviz/pull/58)
- plot_structure_2d() doc str add "multiple structures in single figure example" [`#57`](https://github.com/janosh/pymatviz/pull/57)
- update examples/mp_bimodal_e_form.ipynb with MP r2SCAN beta release formation energies [`396bdf4`](https://github.com/janosh/pymatviz/commit/396bdf40f8202ca51d15e97ac87d2b25ab20fd88)
- add save_fig() to pymatviz/utils.py covered by test_save_fig() [`bcb8a06`](https://github.com/janosh/pymatviz/commit/bcb8a063ed5c0d5bb9aea38c28083d3bfe52be19)
- add assets/make_api_docs.py [`1b05792`](https://github.com/janosh/pymatviz/commit/1b05792fa42fcefdc152b647bafbcce5b2504f3c)
- Revert "Python 3.7 support (#55)" [`4335001`](https://github.com/janosh/pymatviz/commit/4335001effac428736812ed1ba33a30c396ebbb4)
- remove google colab compat notices from example notebooks [`e7cc488`](https://github.com/janosh/pymatviz/commit/e7cc4885022c4515b2aed019a6083d9efcd98865)
- tweak docs CSS, add site footer [`6319379`](https://github.com/janosh/pymatviz/commit/6319379ee02f4185599aa5d95308d3274091662f)
- type test functions, update commit hooks incl. mypy, setup.cfg set implicit_optional=true which defaults to false in newer mypys [`9303f7d`](https://github.com/janosh/pymatviz/commit/9303f7dc0a02d47f981d0bbc63f916fec7bdf1fd)
- add docformatter pre-commit hook [`25d700c`](https://github.com/janosh/pymatviz/commit/25d700c5daac956d40b8626958ed4557d9872110)
- add citation.cff [`531133c`](https://github.com/janosh/pymatviz/commit/531133c3a8ab20251101b42cbc1f6f190b76c546)
- residual_hist() remove args y_true, y_pred, now takes y_res directly [`5c8ccbe`](https://github.com/janosh/pymatviz/commit/5c8ccbe193963b9b56d3773c2f3881ddd9d9d0af)

## [v0.5.1](https://github.com/janosh/pymatviz/compare/v0.5.0...v0.5.1)

> 8 October 2022

- Python 3.7 support [`#55`](https://github.com/janosh/pymatviz/pull/55)
- move plot_defaults.py into pymatviz pkg [`#23973`](https://github.com/matplotlib/matplotlib/issues/23973)
- add kwarg y_max_headroom=float to annotate_bars() [`d613d99`](https://github.com/janosh/pymatviz/commit/d613d999174da7f9750ebfffecffd5418e92a8e8)

## [v0.5.0](https://github.com/janosh/pymatviz/compare/v0.4.4...v0.5.0)

> 21 September 2022

- Support log-scaled heat maps in `ptable_heatmap_plotly()` [`#53`](https://github.com/janosh/pymatviz/pull/53)
- Improve tests for `parity.py` and `relevance.py` [`#51`](https://github.com/janosh/pymatviz/pull/51)
- Fix `plot_structure_2d()` `show_bonds` for disordered structures [`#50`](https://github.com/janosh/pymatviz/pull/50)
- Dataset exploration [`#49`](https://github.com/janosh/pymatviz/pull/49)
- Add `examples/mp_bimodal_e_form.ipynb` [`#47`](https://github.com/janosh/pymatviz/pull/47)
- Add unary ptable heatmap plot to `examples/mprester_ptable.ipynb` [`#46`](https://github.com/janosh/pymatviz/pull/46)
- fix: add .py to readme link [`#45`](https://github.com/janosh/pymatviz/pull/45)
- add test_ptable_heatmap_plotly_kwarg_combos [`#44`](https://github.com/janosh/pymatviz/pull/44)
- Fix windows CI [`#43`](https://github.com/janosh/pymatviz/pull/43)
- rm data/ex-ensemble-roost.csv, generate random regression data to plot example assets with numpy instead [`11a47d3`](https://github.com/janosh/pymatviz/commit/11a47d386cd2194e44cca58e668a108a2753bb14)
- breaking: rename `{elements=>ptable}.py` [`1a87845`](https://github.com/janosh/pymatviz/commit/1a87845f4018848523895e9d997320ca8276a3a4)
- merge `pymatviz/{quantile,ranking}.py` into new pymatviz/uncertainty.py [`c4e0872`](https://github.com/janosh/pymatviz/commit/c4e087227f7053fc5a10a3301f8f5d54cd1d7cb3)
- add examples/mprester_ptable.ipynb [`04bf077`](https://github.com/janosh/pymatviz/commit/04bf0776b57af52ec871cc877c9fd5c4a03c7eb1)
- cleaner plt.Axes and go.Figure imports [`d21e6b4`](https://github.com/janosh/pymatviz/commit/d21e6b400b051cab26890888904bd24bb4f2c901)
- add JupyterDash example to `mprester_ptable.ipynb` [`c4381d3`](https://github.com/janosh/pymatviz/commit/c4381d3242298ed0a9b85c82298037d9b822f734)
- remove save_reference_img() from conftest.py [`73fd600`](https://github.com/janosh/pymatviz/commit/73fd6005d7978a7e3cced81d8846060e81a74235)
- rename `cum_{res,err}` to `cumulative_{residual,error}` [`5d52181`](https://github.com/janosh/pymatviz/commit/5d52181c5af3da9d11755b1b01a19a7dc059961e)
- add readme table with links to open notebooks in GH, Colab, Binder [`13df583`](https://github.com/janosh/pymatviz/commit/13df5831de5dec6dd58166fb1e3cb89869053f83)
- add standardize_struct kwarg to plot_structure_2d() [`21d1264`](https://github.com/janosh/pymatviz/commit/21d1264a3d443ebd7176955a1aef1891b93e98f1)
- add pre-commit hook <https://github.com/jendrikseipp/vulture> [`9dfb806`](https://github.com/janosh/pymatviz/commit/9dfb806e5b929b80f60dd684a764fd71d890df77)

## [v0.4.4](https://github.com/janosh/pymatviz/compare/v0.4.3...v0.4.4)

> 9 July 2022

- Add kwarg `show_bonds` to `plot_structure_2d()` [`#41`](https://github.com/janosh/pymatviz/pull/41)
- Add kwarg `exclude_elements: Sequence[str]` to `ptable_heatmap()` [`#40`](https://github.com/janosh/pymatviz/pull/40)
- fix broken readme image ![matbench-phonons-structures-2d] [`17742d1`](https://github.com/janosh/pymatviz/commit/17742d161151db1f9e1d615e6f19a3eac678be27)
- fix flaky CI error from matminer load_dataset() [`0c0c043`](https://github.com/janosh/pymatviz/commit/0c0c043bde14f356ddfc2335f5bd900e91c159fa)
- fix plot_structure_2d() for pymatgen structures with oxidation states [`b61f4c0`](https://github.com/janosh/pymatviz/commit/b61f4c0931f5bc9eb1f28bbb22973fd5f4b1699a)
- plotly add_identity_line() use fig.full_figure_for_development() to get x/y-range [`8ddf049`](https://github.com/janosh/pymatviz/commit/8ddf049058985fed47ceadde5df62e1222dd2502)
- move codespell and pydocstyle commit hook args to setup.cfg [`7338173`](https://github.com/janosh/pymatviz/commit/7338173df9df9947bf01b5905a08e086e60f2400)
- [pre-commit.ci] pre-commit autoupdate [`a0b5195`](https://github.com/janosh/pymatviz/commit/a0b5195809bb8af36de3f9fb15355fb155efb01f)

## [v0.4.3](https://github.com/janosh/pymatviz/compare/v0.4.2...v0.4.3)

> 2 June 2022

- Fix GH not showing interactive Plotly figures in Jupyter [`#39`](https://github.com/janosh/pymatviz/pull/39)
- Add new plotly function `sankey_from_2_df_cols()` [`#37`](https://github.com/janosh/pymatviz/pull/37)
- note relation to ase.visualize.plot.plot_atoms() in plot_structure_2d() doc string [`f4c9fb7`](https://github.com/janosh/pymatviz/commit/f4c9fb77c2b863a1b735a28e8d6c9b04ae21380f)
- corrections to element property data in pymatviz/elements.csv (thanks @robertwb) + origin link [`444f9ba`](https://github.com/janosh/pymatviz/commit/444f9ba13b92dd28b7b280e19b6df01164dacd81)
- fix add_identity_line() if plotly trace data contains NaNs [`c31ac55`](https://github.com/janosh/pymatviz/commit/c31ac55366e02b65f4614595380589feb4855de6)

## [v0.4.2](https://github.com/janosh/pymatviz/compare/v0.4.1...v0.4.2)

> 16 May 2022

- Improve `ptable_heatmap_plotly()` `colorscale` kwarg [`#35`](https://github.com/janosh/pymatviz/pull/35)
- [pre-commit.ci] pre-commit autoupdate [`#34`](https://github.com/janosh/pymatviz/pull/34)
- Accept pmg structures as input for `spacegroup_(hist|sunburst)` [`#33`](https://github.com/janosh/pymatviz/pull/33)
- Fix spacegroup_hist() crystal system counts [`#32`](https://github.com/janosh/pymatviz/pull/32)
- Expand testing of keyword arguments [`#31`](https://github.com/janosh/pymatviz/pull/31)
- generate_assets.py refactor save_and_compress_svg() [`a47b811`](https://github.com/janosh/pymatviz/commit/a47b811a877a1f9792ff93d967f999d3ef410006)
- add add_identity_line() in tests/test_utils.py [`faa6a52`](https://github.com/janosh/pymatviz/commit/faa6a5275145075ee8d47446e8f5c9eed3075cb9)
- readme remove ml-matrics to pymatviz migration cmd [`0d123d4`](https://github.com/janosh/pymatviz/commit/0d123d4952e96684ee24e02905249867b16ce69f)

## [v0.4.1](https://github.com/janosh/pymatviz/compare/v0.4.0...v0.4.1)

> 20 March 2022

- Support space groups symbols as input to `spacegroup_hist()` [`#30`](https://github.com/janosh/pymatviz/pull/30)
- Support space groups symbols as input to `spacegroup_sunburst()` [`#28`](https://github.com/janosh/pymatviz/pull/28)
- Add some Juypter notebooks with usage examples [`#25`](https://github.com/janosh/pymatviz/pull/25)
- rm local structures data/structures/mp-*.yml, use matbench phonon structures instead [`7ee4105`](https://github.com/janosh/pymatviz/commit/7ee4105da56c5a2df2a4e060a87adf5d09d90dbc)
- GH action bump setup-python to v3 [`9f21079`](https://github.com/janosh/pymatviz/commit/9f21079605a7748a14328f38971f69633fafd628)
- incl jupyter notebooks in ml-matrics to pymatviz migration cmd [`619075f`](https://github.com/janosh/pymatviz/commit/619075f8f2a49edffa2c38b3376736252e55b547)
- readme clarify not being associated with pymatgen [`64c95fb`](https://github.com/janosh/pymatviz/commit/64c95fbf69627419da113dc145de7f005ebbdf56)
- fix plt.tight_layout() throwing AttributeError on density_hexbin plots (<https://github.com/matplotlib/matplotlib/issues/22576>) [`e5599a6`](https://github.com/janosh/pymatviz/commit/e5599a64d1dbf7f3308cc083f4841932b15c2db6)
- compress new SVG assets [`fe88271`](https://github.com/janosh/pymatviz/commit/fe882719dadbd57ee8bfb567b9d850ee4d38baef)

## [v0.4.0](https://github.com/janosh/pymatviz/compare/v0.3.0...v0.4.0)

> 1 March 2022

- Rename package to pymatviz (formerly ml-matrics) [`#23`](https://github.com/janosh/pymatviz/pull/23)

## [v0.3.0](https://github.com/janosh/pymatviz/compare/v0.2.6...v0.3.0)

> 28 February 2022

- Add `plot_structure_2d()` in new module `ml_matrics/struct_vis.py` [`#20`](https://github.com/janosh/pymatviz/pull/20)
- `git mv data/{mp-n_elements<2,mp-elements}.csv` (closes #19) [`#19`](https://github.com/janosh/pymatviz/issues/19)
- support atomic numbers in count_elements(), only element symbols before, add kwarg text_color in ptable_heatmap [`ada57cc`](https://github.com/janosh/pymatviz/commit/ada57cca715e190d322dcad0cb9c9b949fe20211)
- add kwargs `{pre,suf}fix` in `add_mae_r2_box()`, use `pip` cache in `**publish.yml**` [`6f64c3b`](https://github.com/janosh/pymatviz/commit/6f64c3b5191f3effbb28cc5dd891e8211b368f61)
- better handling of atomic numbers in count_elements() when outside range [1, 118] [`e46b2c4`](https://github.com/janosh/pymatviz/commit/e46b2c44b97719013aab9339e7e557d8dc2da0b1)
- python-requires>=3.8 [`e0560af`](https://github.com/janosh/pymatviz/commit/e0560af810b12cae3a34dd818d628360de2d96e5)

## [v0.2.6](https://github.com/janosh/pymatviz/compare/v0.2.5...v0.2.6)

> 6 February 2022

- add test for count_elements [`4060d4e`](https://github.com/janosh/pymatviz/commit/4060d4e00abe4781a0789721145fc29937fc38f5)
- use future import for py 3.10 type annotations [`ae5aa96`](https://github.com/janosh/pymatviz/commit/ae5aa96e4e5f364d0e7fd3694bd9a608989a6d3a)
- fix get_crystal_sys raise ValueError on non-positive space group numbers [`9a535f7`](https://github.com/janosh/pymatviz/commit/9a535f73c1f2652a9cc65d5752957971d0a54ad2)
- add .github/workflows/publish.yml [`7ce1c55`](https://github.com/janosh/pymatviz/commit/7ce1c558b545adc2889e855ab7c990b3451b06d7)
- use `actions/setup-{python,node}` auto caching [`613706c`](https://github.com/janosh/pymatviz/commit/613706c1dea185ff78ca3836319add2d9629c5c1)
- drop py3.7 support, update pre-commit hooks [`93c3eeb`](https://github.com/janosh/pymatviz/commit/93c3eeb74ad316e07628e3b2a462f693498e6a33)
- gha test.yml add pytest-cov [`16c92ab`](https://github.com/janosh/pymatviz/commit/16c92abc121d3aea8279e36172d2da15cf72a9e4)
- readme remove asset compression note, rename scripts [`41d5b6c`](https://github.com/janosh/pymatviz/commit/41d5b6c9c2d392540b39192241b0afabb91f9816)

## [v0.2.5](https://github.com/janosh/pymatviz/compare/v0.2.4...v0.2.5)

> 26 November 2021

- add ptable_heatmap_plotly() (closes #16) [`#16`](https://github.com/janosh/pymatviz/issues/16)
- delete metrics.py module + tests as they're better placed in CompRhys/aviary#13 [`6870f2d`](https://github.com/janosh/pymatviz/commit/6870f2d2a2f13a36e240589651627b3298ae2a02)
- add new plotly fig spacegroup_sunburst [`871c42a`](https://github.com/janosh/pymatviz/commit/871c42a6a5217015288e01bcc9ce83b680a9370a)
- some tweaks to ptable_heatmap_plotly, delete unused softmax + one_hot utils [`3e931f2`](https://github.com/janosh/pymatviz/commit/3e931f2a2a3ef64a0a9529e4a1484479f5efdc68)

## [v0.2.4](https://github.com/janosh/pymatviz/compare/v0.2.3...v0.2.4)

> 2 November 2021

- ptable_heatmap add label precision kwarg, change text color to white on dark tiles, spacegroup_hist better label fontsizes, doc string improvements all around, pre-commit enforce all assets are SVGs [`d45511f`](https://github.com/janosh/pymatviz/commit/d45511f006f03a3054e6daa922a9954e73528080)

## [v0.2.3](https://github.com/janosh/pymatviz/compare/v0.2.2...v0.2.3)

> 18 October 2021

- refactor ptable_heatmap to plot arbitrary data, not just elemental prevalence, add element properties to elements.csv [`57ceb4d`](https://github.com/janosh/pymatviz/commit/57ceb4d547cf7a500b320d8b62440579d64de88b)
- ptable_heatmap add heat_labels kwarg for fraction, percent or None heat labels, make ptable_heatmap_ratio colors and legend customizable [`c3210bf`](https://github.com/janosh/pymatviz/commit/c3210bf9363e927b5a1de6a2e1053284e541353c)

## [v0.2.2](https://github.com/janosh/pymatviz/compare/v0.2.1...v0.2.2)

> 14 October 2021

- add typing_extensions as py37 dep and dynamically import Literal [`e69face`](https://github.com/janosh/pymatviz/commit/e69face983b71dc0c0ec0c4db298ea0161a38022)
- add codespell pre-commit hook + fix typos [`58bfa75`](https://github.com/janosh/pymatviz/commit/58bfa7513dcf6d75e9bba3a3dac4fc8a1a54cd6e)

## [v0.2.1](https://github.com/janosh/pymatviz/compare/v0.2.0...v0.2.1)

> 7 October 2021

- readme move ptable_heatmap() and spacegroup_hist() plots to top [`ac000a4`](https://github.com/janosh/pymatviz/commit/ac000a49c8ea28ca27def139b66b131463f91b48)
- spacegroup_hist() add crystal system counts [`693307e`](https://github.com/janosh/pymatviz/commit/693307efe3a0d2eccd908a8dd702d67eb1cc33b8)
- [pre-commit.ci] pre-commit autoupdate [`d6227dd`](https://github.com/janosh/pymatviz/commit/d6227dd606c171116b742a3a4726d2b09a52e096)

## [v0.2.0](https://github.com/janosh/pymatviz/compare/v0.1.9...v0.2.0)

> 4 October 2021

- doc string improvements [`6572d85`](https://github.com/janosh/pymatviz/commit/6572d8547a9f0f0e451d98bd12e46a1d4f409204)
- rename ptable_elemental_prevalence -> ptable_heatmap, ptable_elemental_ratio -> ptable_heatmap_ratio [`f4c915d`](https://github.com/janosh/pymatviz/commit/f4c915d0b57fb83f5f7047097d1c46a1ec921f51)
- add pydocstyle pre-commit hook [`a379621`](https://github.com/janosh/pymatviz/commit/a37962161da0e8abcce0bfb3252f192a59e46e50)
- err_decay() accept and return axes object [`60c0ceb`](https://github.com/janosh/pymatviz/commit/60c0ceb036872cbbc1e3862ca68ebb994968963c)
- handle nan values in add_mae_r2_box [`be0acf1`](https://github.com/janosh/pymatviz/commit/be0acf1cc66e249afb5e740c24ff5c9c7525157d)
- fix on.paths in CI workflows [`f9e9ba2`](https://github.com/janosh/pymatviz/commit/f9e9ba27fa45bb1516455fa1c0ed2f332005684b)

## [v0.1.9](https://github.com/janosh/pymatviz/compare/v0.1.8...v0.1.9)

> 26 August 2021

- remove ml_matrics.utils.add_identity, use plt.axline instead <https://git.io/JERaj> [`d30a29f`](https://github.com/janosh/pymatviz/commit/d30a29fce537c187bbaf7d447553abb5a7a06af1)

## [v0.1.8](https://github.com/janosh/pymatviz/compare/v0.1.7...v0.1.8)

> 25 August 2021

- [pre-commit.ci] pre-commit autoupdate [`#12`](https://github.com/janosh/pymatviz/pull/12)
- use numpy.typing.NDArray for type hints [`c99062c`](https://github.com/janosh/pymatviz/commit/c99062c307d7a3fbdb3ae8033e148f8327ff7b38)
- add some more pre-commit-hooks [`dfb93e4`](https://github.com/janosh/pymatviz/commit/dfb93e4300c51e439b58033d363b40512a79d877)

## [v0.1.7](https://github.com/janosh/pymatviz/compare/v0.1.6...v0.1.7)

> 3 July 2021

- fully type annotate all functions and fix mypy errors [`b5729e3`](https://github.com/janosh/pymatviz/commit/b5729e363c1c7c30417dd9d1a01616732a06eb36)

## [v0.1.6](https://github.com/janosh/pymatviz/compare/v0.1.5...v0.1.6)

> 2 July 2021

- fix ptable_elemental_prevalence cbar_max kwarg [`9c92e7c`](https://github.com/janosh/pymatviz/commit/9c92e7c49601bec0d93f9ed3350ff69d0c795959)

## [v0.1.5](https://github.com/janosh/pymatviz/compare/v0.1.4...v0.1.5)

> 12 May 2021

- [pre-commit.ci] pre-commit autoupdate [`#11`](https://github.com/janosh/pymatviz/pull/11)
- ptable_elemental_prevalence change color map as black text on dark green (high prevalence) elements was unreadable [`8bc17b5`](https://github.com/janosh/pymatviz/commit/8bc17b5c77b56760151bfb8522ba7766680920ea)

## [v0.1.4](https://github.com/janosh/pymatviz/compare/v0.1.3...v0.1.4)

> 6 May 2021

- add count as label below element symbols in ptable_elemental_prevalence [`1a8d077`](https://github.com/janosh/pymatviz/commit/1a8d077ecb854c10e07fc788ede782b87fedd59b)
- add format-ipy-cells pre-commit hook [`7f83ce3`](https://github.com/janosh/pymatviz/commit/7f83ce3dab3259559cfb5d00d6bd8399c699a5fb)
- [pre-commit.ci] pre-commit autoupdate [`70a5695`](https://github.com/janosh/pymatviz/commit/70a5695d7f48e857f4684dff199f3f7b88ba5c31)

## [v0.1.3](https://github.com/janosh/pymatviz/compare/v0.1.2...v0.1.3)

> 10 April 2021

- ptable_elemental_prevalence add cbar_max kwarg [`829f762`](https://github.com/janosh/pymatviz/commit/829f762eaed18d50b116d5205a2ee9c46ba088d0)

## [v0.1.2](https://github.com/janosh/pymatviz/compare/v0.1.1...v0.1.2)

> 6 April 2021

- release as PyPI package [`1ab0d29`](https://github.com/janosh/pymatviz/commit/1ab0d290d4de465b2c384e5dcc90f80b74d7f12b)
- manually merge branch metrics. thx @CompRhys! [`3ddb232`](https://github.com/janosh/pymatviz/commit/3ddb2325f2416abfa4c75b43201e03ae4589ac18)
- rename repo mlmatrics -> ml-matrics [`b65f50f`](https://github.com/janosh/pymatviz/commit/b65f50f85b2e59ac3233217454e4a76fadbe3238)
- fix ptable_elemental_prevalence log scale colorbar [`0913416`](https://github.com/janosh/pymatviz/commit/091341673f66cab49b97ccaed6ffb0850eb570b2)
- readme use referenced links for raw.githubusercontent images [`cbd1033`](https://github.com/janosh/pymatviz/commit/cbd1033b4a549204c16fe5e32b95182b890206d7)
- rename branch master -> main [`50d73c6`](https://github.com/janosh/pymatviz/commit/50d73c6c43e0e04bd3e0639ff5bc25e0e5e393ea)
- fix setup.cfg to not pollute top level namespace in site-packages with generic data folder [`10c8589`](https://github.com/janosh/pymatviz/commit/10c8589b2c07799f05cbe7986d2dea2c527e7598)

## [v0.1.1](https://github.com/janosh/pymatviz/compare/v0.1.0...v0.1.1)

> 6 April 2021

- fix setup.cfg to not pollute top level namespace in site-packages with generic data folder [`68a84b2`](https://github.com/janosh/pymatviz/commit/68a84b2fa2afbdf6afe21873bbe48b1465e9c20f)
- [pre-commit.ci] pre-commit autoupdate [`49caa7f`](https://github.com/janosh/pymatviz/commit/49caa7fe7a34002b0bfa5d1edb0b31d36bc8639a)

## v0.1.0

> 5 April 2021

- concept names, publication plot params [`#2`](https://github.com/janosh/pymatviz/pull/2)
- clean up cumulative plots, add better data source for example plots, rename density_scatter to parity [`#1`](https://github.com/janosh/pymatviz/pull/1)
- add spacegroup_hist to histograms.py [`41297ec`](https://github.com/janosh/pymatviz/commit/41297ec50cfd16ef2edb908b5e7921bae6916365)
- add clf_metrics.py [`ce4f864`](https://github.com/janosh/pymatviz/commit/ce4f8649fb1b48f8fdc2a921cab3c80887593ba5)
- initial commit [`4a985c1`](https://github.com/janosh/pymatviz/commit/4a985c1f8ebe48749da7da4d0722ebd696ffa0ab)
- add correlation.py with marchenko_pastur plot [`2e6665c`](https://github.com/janosh/pymatviz/commit/2e6665c33680a7d20636858c7f04d652d9d8700c)
- add more tests, especially for kwargs [`600adbf`](https://github.com/janosh/pymatviz/commit/600adbf0965c559af0fdf8c80644c56acb5c6038)
- add residual_vs_actual(), residual_hist() in new histograms.py, add readme glossary [`ed8393c`](https://github.com/janosh/pymatviz/commit/ed8393c0db16e0d90334fe94eaf9d607894b2bbe)
- release as PyPI package [`511b5e3`](https://github.com/janosh/pymatviz/commit/511b5e3c326bcad714596e5559814ca44797a8d4)
- manually merge branch metrics. thx @CompRhys! [`c3c284d`](https://github.com/janosh/pymatviz/commit/c3c284dab501e007953c31aca01c02c025bcb369)
- add bar values percent or count to hist_elemental_prevalence, more tests for elements.py + log_scale plots in readme [`8d722f8`](https://github.com/janosh/pymatviz/commit/8d722f82dfeb736120225a4a4755baa3427de8ec)
- rename repo mlmatrics -> ml-matrics [`d5b741c`](https://github.com/janosh/pymatviz/commit/d5b741c93897a9e23c758b5729d72f5e28636d10)
- change ptable prevalence colors, log now uses base 10 [`494ac24`](https://github.com/janosh/pymatviz/commit/494ac2428795cc1796ca3409fbf66b4984fbc9d9)
- add true_pred_hist to histograms.py [`0ea0ef6`](https://github.com/janosh/pymatviz/commit/0ea0ef65a1712f3d643525c9e15bf62631fa04f0)
- add new plot ptable_elemental_ratio [`6e91792`](https://github.com/janosh/pymatviz/commit/6e917924f40a67db545bb6002ce52c4def8e503f)
- add err_decay to parity.py, display MAE + R^2 on all parity plots [`3999b60`](https://github.com/janosh/pymatviz/commit/3999b601866204a6de956c0342733b4606ea5b12)
- add type hints to parity.py [`ea94b0d`](https://github.com/janosh/pymatviz/commit/ea94b0def714cc4d92f014a3f6cf703d87462e16)
- feature: add regression basic metrics for ensembles [`141f4d9`](https://github.com/janosh/pymatviz/commit/141f4d92b7a3f174cf0634fad1adab6665c11f25)
- fea: allow pre-computed elem_counts for ratio plot, allow cmap to be changed via kwarg [`7ac598b`](https://github.com/janosh/pymatviz/commit/7ac598b73a615a85453ccaad1e95319c86a3989d)
- fix ptable_elemental_prevalence log scale colorbar [`c9d085e`](https://github.com/janosh/pymatviz/commit/c9d085eca04e241b568e30608fa95a6dab5eb24d)
- add err_decay plots + plot function names to readme [`9e994bb`](https://github.com/janosh/pymatviz/commit/9e994bbe38000c744383dc021d1fa3f91c8fcb5b)
- err_decay use analytic rand_mean rather than estimating from sample [`be2b834`](https://github.com/janosh/pymatviz/commit/be2b834fc0800613791135bb86e6896c86c93d6e)
- change ptable plots color bar to show log numbers when log_scale=True [`ed35532`](https://github.com/janosh/pymatviz/commit/ed35532f4ad66f12a19e022e07cfc7aebc8bd48b)
- add doc strings for density_scatter, residual_vs_actual, revert dependence on numpy.typing for now [`006fdff`](https://github.com/janosh/pymatviz/commit/006fdffd3b31fa1e5431db791b55537767f6d660)
- refactor using ax.set property batch setter [`5a41284`](https://github.com/janosh/pymatviz/commit/5a41284adc7d11882e44a8f8c633898cef286e53)
- add marchenko-pastur example with significant eigenvalue [`db7e725`](https://github.com/janosh/pymatviz/commit/db7e72532112d57a86a8a9b2d62fa5b822569117)
- readme add tip for how to depend on this package, regenerate requirements with pipreqs --force . [`cdae757`](https://github.com/janosh/pymatviz/commit/cdae757e731e4e7815aa3b1d29b3c3711daa7f9d)
- fix spacegroup_hist bars having wrong face colors [`6a9ee75`](https://github.com/janosh/pymatviz/commit/6a9ee759f5e301029fbf40e29c84577f2cf8cd8c)
- add github action to compress SVGs on PR [`2be7725`](https://github.com/janosh/pymatviz/commit/2be7725dae369023b5c40c5b560168a478e988b1)
- appease linter [`39a3b15`](https://github.com/janosh/pymatviz/commit/39a3b152fdabfcf98c1c8126283f86afefd19b49)
- speed up count_elements by removing for loop [`1495701`](https://github.com/janosh/pymatviz/commit/14957019e9c1183d050ee6e025d86bc7ab96685e)
- add doc strings to relevance.py [`f372631`](https://github.com/janosh/pymatviz/commit/f372631c6a49427f7ce4d38c3f1ef698c2bedfaa)
- rename branch master -> main [`f56cf6e`](https://github.com/janosh/pymatviz/commit/f56cf6e0be2fcf9f1d423252062d5fea769ceff7)
- fix requirements typo [`9653a82`](https://github.com/janosh/pymatviz/commit/9653a8245feb6107f9d901c42d557a94df1f7517)
- ptable_elemental_prevalence display original values above log-scale colorbar [`665b05c`](https://github.com/janosh/pymatviz/commit/665b05c47a978c769cc16d089274777622b2db62)
- fix qq_gaussian miscal_area [`8239949`](https://github.com/janosh/pymatviz/commit/82399495796340ac58d2f139c72aedb2a908e8ad)
