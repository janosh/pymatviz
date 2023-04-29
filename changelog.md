### Changelog

All notable changes to this project will be documented in this file. Dates are displayed in UTC.

<!-- auto-changelog-above -->

#### [v0.6.2](https://github.com/janosh/pymatviz/compare/v0.6.1...v0.6.2)

> 29 April 2023

- Per-module doc pages [`#77`](https://github.com/janosh/pymatviz/pull/77)
- Refactor `make_docs.py` [`#76`](https://github.com/janosh/pymatviz/pull/76)
- DRY workflows [`#74`](https://github.com/janosh/pymatviz/pull/74)
- More flexible `annotate_metrics` util [`#73`](https://github.com/janosh/pymatviz/pull/73)

#### [v0.6.1](https://github.com/janosh/pymatviz/compare/v0.6.0...v0.6.1)

> 21 March 2023

- Add kwarg `axis: bool | str = "off"` to `plot_structure_2d()` [`#72`](https://github.com/janosh/pymatviz/pull/72)
- Add `ptable_heatmap` `cbar_precision` kwarg [`#70`](https://github.com/janosh/pymatviz/pull/70)
- add changelog.md via auto-changelog [`05da617`](https://github.com/janosh/pymatviz/commit/05da61795d3ecb67026f964be79a36cf0737b760)
- add half-baked /plots and /notebook pages [`ed171ec`](https://github.com/janosh/pymatviz/commit/ed171ec947bfcdbbd5bfda85e2ba6b48e0e35e39)
- add svelte-zoo PrevNext to notebooks pages [`05368c0`](https://github.com/janosh/pymatviz/commit/05368c034f6c2aff572011b93c5b4c722ee7559b)
- add new option 'occurrence' for CountMode = element_composition|fractional_composition|reduced_composition [`bf1604a`](https://github.com/janosh/pymatviz/commit/bf1604a27b2dc80d657ae36a0070d83cba4e2e86)
- refactor ptable_heatmap()'s tick_fmt() and add test for cbar_precision kwarg [`3427e1f`](https://github.com/janosh/pymatviz/commit/3427e1fce2ccd13aacf7b2043d9c10c6375027af)
- plot_structure_2d() in site /api docs [`fcf75de`](https://github.com/janosh/pymatviz/commit/fcf75de255cb3a3fb1555b2b1f7595e0be66043d)

#### [v0.6.0](https://github.com/janosh/pymatviz/compare/v0.5.3...v0.6.0)

> 21 February 2023

- Pyproject [`#69`](https://github.com/janosh/pymatviz/pull/69)

#### [v0.5.3](https://github.com/janosh/pymatviz/compare/v0.5.2...v0.5.3)

> 20 February 2023

- Pyproject [`#69`](https://github.com/janosh/pymatviz/pull/69)
- Add Ruff pre-commit hook [`#68`](https://github.com/janosh/pymatviz/pull/68)
- scatter_density() use x, y args as axis labels if strings [`0f2386a`](https://github.com/janosh/pymatviz/commit/0f2386a826855f6a961e1ad356bcc567ba2c2c88)
- fix util save_and_compress_svg() and update plot_structure_2d() assets [`e0020aa`](https://github.com/janosh/pymatviz/commit/e0020aa25f45ad7dbcad4c8c2cfc8eb5542266d9)
- use redirect in layout.ts instead of ugly DOM href surgery to forward readme links to GH  repo [`7da3c0c`](https://github.com/janosh/pymatviz/commit/7da3c0c51ea43871a1a8ee6d9886506d16a5f601)
- rename add_mae_r2_box() to annotate_mae_r2() [`c550332`](https://github.com/janosh/pymatviz/commit/c550332700ed630ce1aa375fa5c2ba45022ccb10)

#### [v0.5.2](https://github.com/janosh/pymatviz/compare/v0.5.1...v0.5.2)

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

#### [v0.5.1](https://github.com/janosh/pymatviz/compare/v0.5.0...v0.5.1)

> 8 October 2022

- Python 3.7 support [`#55`](https://github.com/janosh/pymatviz/pull/55)
- move plot_defaults.py into pymatviz pkg [`#23973`](https://github.com/matplotlib/matplotlib/issues/23973)
- add kwarg y_max_headroom=float to annotate_bars() [`d613d99`](https://github.com/janosh/pymatviz/commit/d613d999174da7f9750ebfffecffd5418e92a8e8)

#### [v0.5.0](https://github.com/janosh/pymatviz/compare/v0.4.4...v0.5.0)

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

#### [v0.4.4](https://github.com/janosh/pymatviz/compare/v0.4.3...v0.4.4)

> 9 July 2022

- Add kwarg `show_bonds` to `plot_structure_2d()` [`#41`](https://github.com/janosh/pymatviz/pull/41)
- Add kwarg `exclude_elements: Sequence[str]` to `ptable_heatmap()` [`#40`](https://github.com/janosh/pymatviz/pull/40)
- fix broken readme image ![matbench-phonons-structures-2d] [`17742d1`](https://github.com/janosh/pymatviz/commit/17742d161151db1f9e1d615e6f19a3eac678be27)
- fix flaky CI error from matminer load_dataset() [`0c0c043`](https://github.com/janosh/pymatviz/commit/0c0c043bde14f356ddfc2335f5bd900e91c159fa)
- fix plot_structure_2d() for pymatgen structures with oxidation states [`b61f4c0`](https://github.com/janosh/pymatviz/commit/b61f4c0931f5bc9eb1f28bbb22973fd5f4b1699a)
- plotly add_identity_line() use fig.full_figure_for_development() to get x/y-range [`8ddf049`](https://github.com/janosh/pymatviz/commit/8ddf049058985fed47ceadde5df62e1222dd2502)
- move codespell and pydocstyle commit hook args to setup.cfg [`7338173`](https://github.com/janosh/pymatviz/commit/7338173df9df9947bf01b5905a08e086e60f2400)
- [pre-commit.ci] pre-commit autoupdate [`a0b5195`](https://github.com/janosh/pymatviz/commit/a0b5195809bb8af36de3f9fb15355fb155efb01f)

#### [v0.4.3](https://github.com/janosh/pymatviz/compare/v0.4.2...v0.4.3)

> 2 June 2022

- Fix GH not showing interactive Plotly figures in Jupyter [`#39`](https://github.com/janosh/pymatviz/pull/39)
- Add new plotly function `sankey_from_2_df_cols()` [`#37`](https://github.com/janosh/pymatviz/pull/37)
- note relation to ase.visualize.plot.plot_atoms() in plot_structure_2d() doc string [`f4c9fb7`](https://github.com/janosh/pymatviz/commit/f4c9fb77c2b863a1b735a28e8d6c9b04ae21380f)
- corrections to element property data in pymatviz/elements.csv (thanks @robertwb) + origin link [`444f9ba`](https://github.com/janosh/pymatviz/commit/444f9ba13b92dd28b7b280e19b6df01164dacd81)
- fix add_identity_line() if plotly trace data contains NaNs [`c31ac55`](https://github.com/janosh/pymatviz/commit/c31ac55366e02b65f4614595380589feb4855de6)

#### [v0.4.2](https://github.com/janosh/pymatviz/compare/v0.4.1...v0.4.2)

> 16 May 2022

- Improve `ptable_heatmap_plotly()` `colorscale` kwarg [`#35`](https://github.com/janosh/pymatviz/pull/35)
- [pre-commit.ci] pre-commit autoupdate [`#34`](https://github.com/janosh/pymatviz/pull/34)
- Accept pmg structures as input for `spacegroup_(hist|sunburst)` [`#33`](https://github.com/janosh/pymatviz/pull/33)
- Fix spacegroup_hist() crystal system counts [`#32`](https://github.com/janosh/pymatviz/pull/32)
- Expand testing of keyword arguments [`#31`](https://github.com/janosh/pymatviz/pull/31)
- generate_assets.py refactor save_and_compress_svg() [`a47b811`](https://github.com/janosh/pymatviz/commit/a47b811a877a1f9792ff93d967f999d3ef410006)
- add add_identity_line() in tests/test_utils.py [`faa6a52`](https://github.com/janosh/pymatviz/commit/faa6a5275145075ee8d47446e8f5c9eed3075cb9)
- readme remove ml-matrics to pymatviz migration cmd [`0d123d4`](https://github.com/janosh/pymatviz/commit/0d123d4952e96684ee24e02905249867b16ce69f)

#### [v0.4.1](https://github.com/janosh/pymatviz/compare/v0.4.0...v0.4.1)

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

#### [v0.4.0](https://github.com/janosh/pymatviz/compare/v0.3.0...v0.4.0)

> 1 March 2022

- Rename package to pymatviz (formerly ml-matrics) [`#23`](https://github.com/janosh/pymatviz/pull/23)

#### [v0.3.0](https://github.com/janosh/pymatviz/compare/v0.2.6...v0.3.0)

> 28 February 2022

- Add `plot_structure_2d()` in new module `ml_matrics/struct_vis.py` [`#20`](https://github.com/janosh/pymatviz/pull/20)
- `git mv data/{mp-n_elements<2,mp-elements}.csv` (closes #19) [`#19`](https://github.com/janosh/pymatviz/issues/19)
- support atomic numbers in count_elements(), only element symbols before, add kwarg text_color in ptable_heatmap [`ada57cc`](https://github.com/janosh/pymatviz/commit/ada57cca715e190d322dcad0cb9c9b949fe20211)
- add kwargs `{pre,suf}fix` in `add_mae_r2_box()`, use `pip` cache in `**publish.yml**` [`6f64c3b`](https://github.com/janosh/pymatviz/commit/6f64c3b5191f3effbb28cc5dd891e8211b368f61)
- better handling of atomic numbers in count_elements() when outside range [1, 118] [`e46b2c4`](https://github.com/janosh/pymatviz/commit/e46b2c44b97719013aab9339e7e557d8dc2da0b1)
- python-requires>=3.8 [`e0560af`](https://github.com/janosh/pymatviz/commit/e0560af810b12cae3a34dd818d628360de2d96e5)

#### [v0.2.6](https://github.com/janosh/pymatviz/compare/v0.2.5...v0.2.6)

> 6 February 2022

- add test for count_elements [`4060d4e`](https://github.com/janosh/pymatviz/commit/4060d4e00abe4781a0789721145fc29937fc38f5)
- use future import for py 3.10 type annotations [`ae5aa96`](https://github.com/janosh/pymatviz/commit/ae5aa96e4e5f364d0e7fd3694bd9a608989a6d3a)
- fix get_crystal_sys raise ValueError on non-positive space group numbers [`9a535f7`](https://github.com/janosh/pymatviz/commit/9a535f73c1f2652a9cc65d5752957971d0a54ad2)
- add .github/workflows/publish.yml [`7ce1c55`](https://github.com/janosh/pymatviz/commit/7ce1c558b545adc2889e855ab7c990b3451b06d7)
- use `actions/setup-{python,node}` auto caching [`613706c`](https://github.com/janosh/pymatviz/commit/613706c1dea185ff78ca3836319add2d9629c5c1)
- drop py3.7 support, update pre-commit hooks [`93c3eeb`](https://github.com/janosh/pymatviz/commit/93c3eeb74ad316e07628e3b2a462f693498e6a33)
- gha test.yml add pytest-cov [`16c92ab`](https://github.com/janosh/pymatviz/commit/16c92abc121d3aea8279e36172d2da15cf72a9e4)
- readme remove asset compression note, rename scripts [`41d5b6c`](https://github.com/janosh/pymatviz/commit/41d5b6c9c2d392540b39192241b0afabb91f9816)

#### [v0.2.5](https://github.com/janosh/pymatviz/compare/v0.2.4...v0.2.5)

> 26 November 2021

- add ptable_heatmap_plotly() (closes #16) [`#16`](https://github.com/janosh/pymatviz/issues/16)
- delete metrics.py module + tests as they're better placed in CompRhys/aviary#13 [`6870f2d`](https://github.com/janosh/pymatviz/commit/6870f2d2a2f13a36e240589651627b3298ae2a02)
- add new plotly fig spacegroup_sunburst [`871c42a`](https://github.com/janosh/pymatviz/commit/871c42a6a5217015288e01bcc9ce83b680a9370a)
- some tweaks to ptable_heatmap_plotly, delete unused softmax + one_hot utils [`3e931f2`](https://github.com/janosh/pymatviz/commit/3e931f2a2a3ef64a0a9529e4a1484479f5efdc68)

#### [v0.2.4](https://github.com/janosh/pymatviz/compare/v0.2.3...v0.2.4)

> 2 November 2021

- ptable_heatmap add label precision kwarg, change text color to white on dark tiles, spacegroup_hist better label fontsizes, doc string improvements all around, pre-commit enforce all assets are SVGs [`d45511f`](https://github.com/janosh/pymatviz/commit/d45511f006f03a3054e6daa922a9954e73528080)

#### [v0.2.3](https://github.com/janosh/pymatviz/compare/v0.2.2...v0.2.3)

> 18 October 2021

- refactor ptable_heatmap to plot arbitrary data, not just elemental prevalence, add element properties to elements.csv [`57ceb4d`](https://github.com/janosh/pymatviz/commit/57ceb4d547cf7a500b320d8b62440579d64de88b)
- ptable_heatmap add heat_labels kwarg for fraction, percent or None heat labels, make ptable_heatmap_ratio colors and legend customizable [`c3210bf`](https://github.com/janosh/pymatviz/commit/c3210bf9363e927b5a1de6a2e1053284e541353c)

#### [v0.2.2](https://github.com/janosh/pymatviz/compare/v0.2.1...v0.2.2)

> 14 October 2021

- add typing_extensions as py37 dep and dynamically import Literal [`e69face`](https://github.com/janosh/pymatviz/commit/e69face983b71dc0c0ec0c4db298ea0161a38022)
- add codespell pre-commit hook + fix typos [`58bfa75`](https://github.com/janosh/pymatviz/commit/58bfa7513dcf6d75e9bba3a3dac4fc8a1a54cd6e)

#### [v0.2.1](https://github.com/janosh/pymatviz/compare/v0.2.0...v0.2.1)

> 7 October 2021

- readme move ptable_heatmap() and spacegroup_hist() plots to top [`ac000a4`](https://github.com/janosh/pymatviz/commit/ac000a49c8ea28ca27def139b66b131463f91b48)
- spacegroup_hist() add crystal system counts [`693307e`](https://github.com/janosh/pymatviz/commit/693307efe3a0d2eccd908a8dd702d67eb1cc33b8)
- [pre-commit.ci] pre-commit autoupdate [`d6227dd`](https://github.com/janosh/pymatviz/commit/d6227dd606c171116b742a3a4726d2b09a52e096)

#### [v0.2.0](https://github.com/janosh/pymatviz/compare/v0.1.9...v0.2.0)

> 4 October 2021

- doc string improvements [`6572d85`](https://github.com/janosh/pymatviz/commit/6572d8547a9f0f0e451d98bd12e46a1d4f409204)
- rename ptable_elemental_prevalence -> ptable_heatmap, ptable_elemental_ratio -> ptable_heatmap_ratio [`f4c915d`](https://github.com/janosh/pymatviz/commit/f4c915d0b57fb83f5f7047097d1c46a1ec921f51)
- add pydocstyle pre-commit hook [`a379621`](https://github.com/janosh/pymatviz/commit/a37962161da0e8abcce0bfb3252f192a59e46e50)
- err_decay() accept and return axes object [`60c0ceb`](https://github.com/janosh/pymatviz/commit/60c0ceb036872cbbc1e3862ca68ebb994968963c)
- handle nan values in add_mae_r2_box [`be0acf1`](https://github.com/janosh/pymatviz/commit/be0acf1cc66e249afb5e740c24ff5c9c7525157d)
- fix on.paths in CI workflows [`f9e9ba2`](https://github.com/janosh/pymatviz/commit/f9e9ba27fa45bb1516455fa1c0ed2f332005684b)

#### [v0.1.9](https://github.com/janosh/pymatviz/compare/v0.1.8...v0.1.9)

> 26 August 2021

- remove ml_matrics.utils.add_identity, use plt.axline instead <https://git.io/JERaj> [`d30a29f`](https://github.com/janosh/pymatviz/commit/d30a29fce537c187bbaf7d447553abb5a7a06af1)

#### [v0.1.8](https://github.com/janosh/pymatviz/compare/v0.1.7...v0.1.8)

> 25 August 2021

- [pre-commit.ci] pre-commit autoupdate [`#12`](https://github.com/janosh/pymatviz/pull/12)
- use numpy.typing.NDArray for type hints [`c99062c`](https://github.com/janosh/pymatviz/commit/c99062c307d7a3fbdb3ae8033e148f8327ff7b38)
- add some more pre-commit-hooks [`dfb93e4`](https://github.com/janosh/pymatviz/commit/dfb93e4300c51e439b58033d363b40512a79d877)

#### [v0.1.7](https://github.com/janosh/pymatviz/compare/v0.1.6...v0.1.7)

> 3 July 2021

- fully type annotate all functions and fix mypy errors [`b5729e3`](https://github.com/janosh/pymatviz/commit/b5729e363c1c7c30417dd9d1a01616732a06eb36)

#### [v0.1.6](https://github.com/janosh/pymatviz/compare/v0.1.5...v0.1.6)

> 2 July 2021

- fix ptable_elemental_prevalence cbar_max kwarg [`9c92e7c`](https://github.com/janosh/pymatviz/commit/9c92e7c49601bec0d93f9ed3350ff69d0c795959)

#### [v0.1.5](https://github.com/janosh/pymatviz/compare/v0.1.4...v0.1.5)

> 12 May 2021

- [pre-commit.ci] pre-commit autoupdate [`#11`](https://github.com/janosh/pymatviz/pull/11)
- ptable_elemental_prevalence change color map as black text on dark green (high prevalence) elements was unreadable [`8bc17b5`](https://github.com/janosh/pymatviz/commit/8bc17b5c77b56760151bfb8522ba7766680920ea)

#### [v0.1.4](https://github.com/janosh/pymatviz/compare/v0.1.3...v0.1.4)

> 6 May 2021

- add count as label below element symbols in ptable_elemental_prevalence [`1a8d077`](https://github.com/janosh/pymatviz/commit/1a8d077ecb854c10e07fc788ede782b87fedd59b)
- add format-ipy-cells pre-commit hook [`7f83ce3`](https://github.com/janosh/pymatviz/commit/7f83ce3dab3259559cfb5d00d6bd8399c699a5fb)
- [pre-commit.ci] pre-commit autoupdate [`70a5695`](https://github.com/janosh/pymatviz/commit/70a5695d7f48e857f4684dff199f3f7b88ba5c31)

#### [v0.1.3](https://github.com/janosh/pymatviz/compare/v0.1.2...v0.1.3)

> 10 April 2021

- ptable_elemental_prevalence add cbar_max kwarg [`829f762`](https://github.com/janosh/pymatviz/commit/829f762eaed18d50b116d5205a2ee9c46ba088d0)

#### [v0.1.2](https://github.com/janosh/pymatviz/compare/v0.1.1...v0.1.2)

> 6 April 2021

- release as PyPI package [`1ab0d29`](https://github.com/janosh/pymatviz/commit/1ab0d290d4de465b2c384e5dcc90f80b74d7f12b)
- manually merge branch metrics. thx @CompRhys! [`3ddb232`](https://github.com/janosh/pymatviz/commit/3ddb2325f2416abfa4c75b43201e03ae4589ac18)
- rename repo mlmatrics -> ml-matrics [`b65f50f`](https://github.com/janosh/pymatviz/commit/b65f50f85b2e59ac3233217454e4a76fadbe3238)
- fix ptable_elemental_prevalence log scale colorbar [`0913416`](https://github.com/janosh/pymatviz/commit/091341673f66cab49b97ccaed6ffb0850eb570b2)
- readme use referenced links for raw.githubusercontent images [`cbd1033`](https://github.com/janosh/pymatviz/commit/cbd1033b4a549204c16fe5e32b95182b890206d7)
- rename branch master -> main [`50d73c6`](https://github.com/janosh/pymatviz/commit/50d73c6c43e0e04bd3e0639ff5bc25e0e5e393ea)
- fix setup.cfg to not pollute top level namespace in site-packages with generic data folder [`10c8589`](https://github.com/janosh/pymatviz/commit/10c8589b2c07799f05cbe7986d2dea2c527e7598)

#### [v0.1.1](https://github.com/janosh/pymatviz/compare/v0.1.0...v0.1.1)

> 6 April 2021

- fix setup.cfg to not pollute top level namespace in site-packages with generic data folder [`68a84b2`](https://github.com/janosh/pymatviz/commit/68a84b2fa2afbdf6afe21873bbe48b1465e9c20f)
- [pre-commit.ci] pre-commit autoupdate [`49caa7f`](https://github.com/janosh/pymatviz/commit/49caa7fe7a34002b0bfa5d1edb0b31d36bc8639a)

#### v0.1.0

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
