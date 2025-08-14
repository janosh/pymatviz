"""Periodic table plots powered by plotly."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.core import Element

from pymatviz.colors import ELEM_TYPE_COLORS
from pymatviz.enums import ElemCountMode
from pymatviz.process_data import count_elements
from pymatviz.typing import (
    VALID_COLOR_ELEM_STRATEGIES,
    ColorElemTypeStrategy,
    ElemValues,
)
from pymatviz.utils import df_ptable
from pymatviz.utils.data import si_fmt
from pymatviz.utils.plotting import luminance


if TYPE_CHECKING:
    from typing import Literal


ColorScale: TypeAlias = (
    str | Sequence[str] | Sequence[tuple[float, str]] | Callable[[str, float, int], str]
)


def ptable_heatmap_plotly(
    values: ElemValues,
    *,
    count_mode: ElemCountMode = ElemCountMode.composition,
    colorscale: str | Sequence[str] | Sequence[tuple[float, str]] = "viridis",
    show_scale: bool = True,
    show_values: bool = True,
    heat_mode: Literal["value", "fraction", "percent"] = "value",
    fmt: str | Callable[[float], str] | None = None,
    hover_props: Sequence[str] | dict[str, str] | None = None,
    hover_data: dict[str, str | int | float] | pd.Series | None = None,
    font_colors: Sequence[str] = (),
    gap: float = 5,
    font_size: int | None = None,
    bg_color: str | None = None,
    nan_color: str = "#eff",
    colorbar: dict[str, Any] | None = None,
    cscale_range: tuple[float | None, float | None] = (None, None),
    exclude_elements: Sequence[str] = (),
    log: bool = False,
    fill_value: float | None = None,
    element_symbol_map: dict[str, str] | None = None,
    label_map: dict[str, str] | Callable[[str], str] | Literal[False] | None = None,
    border: dict[str, Any] | None | Literal[False] = None,
    scale: float = 1.0,
    **kwargs: Any,
) -> go.Figure:
    """Create a Plotly figure with an interactive heatmap of the periodic table.
    Supports hover tooltips with custom data or atomic reference data like
    electronegativity, atomic_radius, etc. See kwargs hover_data and hover_props, resp.

    Args:
        values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values e.g. dict(Fe=2, O=3) or iterable of composition
            strings or Pymatgen composition objects.
        count_mode ("composition" | "fractional_composition" | "reduced_composition"):
            Reduce or normalize compositions before counting. See `count_elements` for
            details. Only used when values is list of composition strings/objects.
        colorscale (str | list[str] | list[tuple[float, str]]): Color scale for heatmap.
            Defaults to "viridis". See plotly.com/python/builtin-colorscales for names
            of other builtin color scales. Note "YlGn" and px.colors.sequential.YlGn are
            equivalent. Custom scales are specified as ["blue", "red"] or
            [[0, "rgb(0,0,255)"], [0.5, "rgb(0,255,0)"], [1, "rgb(255,0,0)"]].
        show_scale (bool): Whether to show a bar for the color scale. Defaults to True.
        show_values (bool): Whether to show numbers on heatmap tiles. Defaults to True.
        heat_mode ("value" | "fraction" | "percent" | None): Whether to display heat
            values as is (value), normalized as a fraction of the total, as percentages
            or not at all (None). Defaults to "value".
            "fraction" and "percent" can be used to make the colors in different
            periodic table heatmap plots comparable.
        fmt (str | Callable[[float], str] | None): f-string format option for heat
            labels. Defaults to ".1%" if heat_mode="percent" else ".3g".
        hover_props (list[str] | dict[str, str]): Elemental properties to display in the
            hover tooltip. Can be a list of property names to display only the values
            themselves or a dict mapping names to what they should display as. E.g.
            dict(atomic_mass="atomic weight") will display as `"atomic weight = {x}"`.
            Defaults to None.
            Available properties are: symbol, row, column, name,
            atomic_number, atomic_mass, n_neutrons, n_protons, n_electrons, period,
            group, phase, radioactive, natural, metal, nonmetal, metalloid, type,
            atomic_radius, electronegativity, first_ionization, density, melting_point,
            boiling_point, number_of_isotopes, discoverer, year, specific_heat,
            n_shells, n_valence.
        hover_data (dict[str, str | int | float] | pd.Series): Map from element symbols
            to additional data to display in the hover tooltip. dict(Fe="this appears in
            the hover tooltip on a new line below the element name"). Defaults to None.
        font_colors (list[str]): One color name or two for [min_color, max_color].
            min_color is applied to annotations with heatmap values less than
            (max_val - min_val) / 2. Defaults to None, meaning auto-set to maximize
            contrast with color scale: white text for dark background and vice versa.
            swapped depending on the colorscale.
        gap (float): Gap in pixels between tiles of the periodic table. Defaults to 5.
        font_size (int): Element symbol and heat label text size. Any valid CSS size
            allowed. Defaults to automatic font size based on plot size. Element symbols
            will be bold and 1.5x this size.
        bg_color (str): Plot background color. Defaults to "rgba(0, 0, 0, 0)".
        colorbar (dict[str, Any]): Plotly colorbar properties documented at
            https://plotly.com/python/reference#heatmap-colorbar. Defaults to
            dict(orientation="h"). Commonly used keys are:
            - title: colorbar title
            - title_side: "top" | "bottom" | "right" | "left"
            - tickmode: "array" | "auto" | "linear" | "log" | "date" | "category"
            - tickvals: list of tick values
            - ticktext: list of tick labels
            - tickformat: f-string format option for tick labels
            - len: fraction of plot height or width depending on orientation
            - thickness: fraction of plot height or width depending on orientation
        nan_color (str): Fill color for element tiles with NaN values. Defaults to
            "#eff".
        cscale_range (tuple[float | None, float | None]): Colorbar range. Defaults to
            (None, None) meaning the range is automatically determined from the data.
        exclude_elements (list[str]): Elements to exclude from the heatmap. E.g. if
            oxygen overpowers everything, you can do exclude_elements=["O"].
            Defaults to ().
        log (bool): Whether to use a logarithmic color scale. Defaults to False.
            Piece of advice: colorscale="viridis" and log=True go well together.
        fill_value (float | None): Value to fill in for missing elements. Defaults to 0.
        element_symbol_map (dict[str, str] | None): A dictionary to map element symbols
            to custom strings. If provided, these custom strings will be displayed
            instead of the standard element symbols. Defaults to None.
        label_map (dict[str, str] | Callable[[str], str] | None): Map heat values (after
            string formatting) to target strings. Set to False to disable. Defaults to
            dict.fromkeys((np.nan, None, "nan", "nan%"), "-") so as not to display "nan"
            for missing values.
        border (dict[str, Any]): Border properties for element tiles. Defaults to
            dict(width=1, color="gray"). Other allowed keys are arguments of go.Heatmap
            which is (mis-)used to draw the borders as a 2nd heatmap below the main one.
            Pass False to disable borders.
        scale (float): Scaling factor for whole figure layout. Defaults to 1.
        **kwargs: Additional keyword arguments passed to
            plotly.figure_factory.create_annotated_heatmap().

    Returns:
        Figure: Plotly Figure object.
    """
    if log and heat_mode in ("fraction", "percent"):
        raise ValueError(
            "Combining log color scale and heat_mode='fraction'/'percent' unsupported"
        )
    if len(cscale_range) != 2:
        raise ValueError(f"{cscale_range=} should have length 2")

    if isinstance(colorscale, str | type(None)):
        colorscale = px.colors.get_colorscale(colorscale or "viridis")
    elif not isinstance(colorscale, Sequence) or not isinstance(
        colorscale[0], str | list | tuple
    ):
        raise TypeError(
            f"{colorscale=} should be string, list of strings or list of "
            "tuples(float, str)"
        )

    colorbar = colorbar or {}
    colorbar.setdefault("orientation", "h")
    # if values is a series with a name, use it as the colorbar title
    if isinstance(values, pd.Series) and values.name:
        colorbar.setdefault("title", values.name)

    values = count_elements(values, count_mode, exclude_elements, fill_value)

    if heat_mode in ("fraction", "percent"):
        # normalize heat values
        clean_vals = values.replace([np.inf, -np.inf], np.nan).dropna()
        # ignore inf values in sum() else all would be set to 0 by normalizing
        heat_value_element_map = values / clean_vals.sum()
    else:
        heat_value_element_map = values

    n_rows, n_columns = 10, 18
    # initialize tile text and hover tooltips to empty strings
    tile_texts, hover_texts = np.full([2, n_rows, n_columns], "", dtype=object)
    heatmap_values = np.full([n_rows, n_columns], np.nan)

    if label_map is None:
        # default to space string for None, np.nan and "nan". space is needed
        # for <br> in tile_text to work so all element symbols are vertically aligned
        label_map = dict.fromkeys([np.nan, None, "nan", "nan%"], "-")  # type: ignore[list-item]

    all_ints = all(isinstance(val, int) for val in values)
    counts_total = values.sum()
    for symbol, period, group, name, *_ in df_ptable.itertuples():
        # build table from bottom up so that period 1 becomes top row
        row = n_rows - period
        col = group - 1

        label = ""  # label (if not None) is placed below the element symbol
        if show_values:
            if symbol in exclude_elements:
                label = "excl."
            elif heat_val := heat_value_element_map.get(symbol):
                if callable(fmt):
                    label = fmt(heat_val)
                elif heat_mode == "percent":
                    label = f"{heat_val:{fmt or '.1%'}}"
                else:
                    fmt_str = fmt if isinstance(fmt, str) else ".1f"
                    label = f"{si_fmt(heat_val, fmt=fmt_str)}".replace("e+0", "e")

            if callable(label_map):
                label = label_map(label)
            elif isinstance(label_map, dict):
                label = label_map.get(label, label)
        # Apply custom element symbol if provided
        display_symbol = (element_symbol_map or {}).get(symbol, symbol)

        style = f"font-weight: bold; font-size: {1.5 * (font_size or 12) * scale};"
        tile_text = f"<span {style=}>{display_symbol}</span>"
        if show_values and label:
            tile_text += f"<br>{label}"

        tile_texts[row][col] = tile_text

        hover_text = f"{name} ({symbol})"

        if heat_val := heat_value_element_map.get(symbol):
            if all_ints:
                # if all values are integers, values are likely element
                # counts, so makes sense to show count and percentage
                percentage = heat_val / counts_total
                hover_text += f"<br>Value: {heat_val} ({percentage:.2%})"
            elif heat_mode == "value":
                hover_text += f"<br>Value: {heat_val:.3g}"
            elif heat_mode in ("fraction", "percent") and (orig_val := values[symbol]):
                hover_text += f"<br>Percentage: {heat_val:.2%} ({orig_val:.3g})"

        if hover_data is not None and symbol in hover_data:
            hover_text += f"<br>{hover_data[symbol]}"

        if hover_props is not None:
            if unsupported_keys := set(hover_props) - set(df_ptable):
                raise ValueError(
                    f"Unsupported hover_props: {', '.join(unsupported_keys)}. Available"
                    f" keys are: {', '.join(df_ptable)}.\nNote that some keys have "
                    "missing values."
                )
            df_row = df_ptable.loc[symbol]
            if isinstance(hover_props, dict):
                for col_name, col_label in hover_props.items():
                    hover_text += f"<br>{col_label} = {df_row[col_name]}"
            elif isinstance(hover_props, list | tuple):
                hover_text += "<br>" + "<br>".join(
                    f"{col_name} = {df_row[col_name]}" for col_name in hover_props
                )
            else:
                raise ValueError(
                    f"hover_props must be dict or sequence of str, got {hover_props}"
                )

        hover_texts[row][col] = hover_text

        if symbol in exclude_elements:
            continue

        color_val = heat_value_element_map[symbol]
        if log and color_val > 0:
            color_val = np.log10(color_val)
        # until https://github.com/plotly/plotly.js/issues/975 is resolved, we need to
        # insert transparency (rgba0) at low end of colorscale (+1e-6) to not show any
        # colors on empty tiles of the periodic table
        heatmap_values[row][col] = color_val

    if isinstance(font_colors, str):
        font_colors = [font_colors]

    non_nan_values = [val for val in heatmap_values.flat if not np.isnan(val)]

    zmin = min(non_nan_values) if cscale_range[0] is None else cscale_range[0]
    zmax = max(non_nan_values) if cscale_range[1] is None else cscale_range[1]
    car_multiplier = 100 if heat_mode == "percent" else 1

    import plotly.figure_factory as ff  # slow import

    cbar_settings = _get_colorbar_settings(colorbar, font_size=font_size, scale=scale)

    fig = ff.create_annotated_heatmap(
        car_multiplier * heatmap_values,
        annotation_text=tile_texts,
        text=hover_texts,
        showscale=show_scale,
        colorscale=colorscale,
        font_colors=font_colors or None,
        hoverinfo="text",
        xgap=gap,
        ygap=gap,
        zauto=False,  # Disable auto-scaling
        zmin=zmin * car_multiplier,
        zmax=zmax * car_multiplier,
        colorbar=cbar_settings,
        **kwargs,
    )

    # Add border heatmap
    if border is not False:
        border = border or {}
        border_color = border.pop("color", "darkgray")
        border_width = border.pop("width", 0.5)

        common_kwargs = dict(
            z=np.where(tile_texts, 1, np.nan), showscale=False, hoverinfo="none"
        )
        # misuse heatmap to add borders around all element tiles
        # 1st one adds the fill color for NaN element tiles, 2nd one adds the border
        fig.add_heatmap(
            **common_kwargs,
            colorscale=[nan_color, nan_color],
            xgap=gap,
            ygap=gap,
        )
        fig.add_heatmap(
            **common_kwargs,
            colorscale=[border_color, border_color],
            xgap=gap - border_width,
            ygap=gap - border_width,
            **border,
        )

        # reverse fig.data to place the border heatmap below the main heatmap
        fig.data = fig.data[::-1]

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10, pad=10),
        paper_bgcolor=bg_color,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(zeroline=False, showgrid=False),
        yaxis=dict(zeroline=False, showgrid=False, scaleanchor="x"),
        font_size=(font_size or 12) * scale,
        width=900 * scale,
        height=500 * scale,
        title=dict(x=0.4, y=0.95),
    )

    horizontal_cbar = colorbar.get("orientation") == "h"
    if horizontal_cbar:
        defaults = dict(
            x=0.4,
            y=0.72,
            title_side="top",
            len=0.4,
            title_font_size=scale * 1.2 * (font_size or 12),
        )
        colorbar = defaults | colorbar
    else:  # make title vertical
        defaults = dict(title_side="right", len=0.87)
        colorbar = defaults | colorbar

    if title := colorbar.get("title"):
        # <br> to increase title standoff
        colorbar["title"] = f"{title}" if horizontal_cbar else f"<br><br>{title}"

    if log:
        orig_min = np.floor(min(non_nan_values))
        orig_max = np.ceil(max(non_nan_values))
        tick_values = np.logspace(orig_min, orig_max, num=10, endpoint=True)
        tick_values = [round(val, -int(np.floor(np.log10(val)))) for val in tick_values]

        colorbar = dict(
            tickvals=np.log10(tick_values),
            ticktext=[si_fmt(v * car_multiplier, fmt=".0f") for v in tick_values],
            **colorbar,
        )

    # suffix % to colorbar title if heat_mode is "percent"
    if heat_mode == "percent" and (cbar_title := colorbar.get("title")):
        colorbar["title"] = f"{cbar_title} (%)"

    cbar_settings = _get_colorbar_settings(colorbar, font_size=font_size, scale=scale)

    if show_scale:
        fig.update_traces(colorbar=cbar_settings)

    return fig


def ptable_hists_plotly(
    data: pd.DataFrame | pd.Series | dict[str, list[float]],
    *,
    # Histogram-specific
    bins: int = 20,
    x_range: tuple[float | None, float | None] | None = None,
    log: bool = False,
    colorscale: str = "RdBu",
    colorbar: dict[str, Any] | Literal[False] | None = None,
    hide_f_block: bool | Literal["auto"] = False,
    # Layout
    font_size: int | None = None,
    scale: float = 1.0,
    # Symbol
    element_symbol_map: dict[str, str] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
    # Annotation
    annotations: dict[str, str | dict[str, Any]]
    | Callable[[Sequence[float]], str | dict[str, Any] | list[dict[str, Any]]]
    | None = None,
    # Element type colors
    color_elem_strategy: ColorElemTypeStrategy = "background",
    elem_type_colors: dict[str, str] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    x_axis_kwargs: dict[str, Any] | None = None,
    y_axis_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Plotly figure with histograms for each element laid out in a periodic table.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[float]]): Map from element
            symbols to histogram values. E.g. if dict, {"Fe": [1, 2, 3], "O": [4, 5]}.
            If pd.Series, index is element symbols and values lists. If pd.DataFrame,
            column names are element symbols histograms are plotted from each column.

        --- Histogram-specific ---
        bins (int): Number of bins for the histograms. Defaults to 20.
        x_range (tuple[float | None, float | None]): x-axis range for all histograms.
            Defaults to None.
        log (bool): Whether to log scale y-axis of each histogram. Defaults to False.
        colorscale (str): Color scale for histogram bars. Defaults to "RdBu" (red to
            blue). See plotly.com/python/builtin-colorscales for other options.
        colorbar (dict[str, Any] | None): Plotly colorbar properties. Defaults to
            dict(orientation="h"). See https://plotly.com/python/reference#heatmap-colorbar
            for available options. Set to False to hide the colorbar.
        hide_f_block (bool | "auto"): Whether to hide the F-block elements. If "auto",
            hide F-block elements if row 7 is empty. Defaults to False.

        --- Layout ---
        font_size (int): Element symbol and annotation text size. Defaults to automatic
            font size based on plot size.
        scale (float): Scaling factor for whole figure layout. Defaults to 1.

        --- Text ---
        element_symbol_map (dict[str, str] | None): A dictionary to map element symbols
            to custom strings. If provided, these custom strings will be displayed
            instead of the standard element symbols. Defaults to None.
        symbol_kwargs (dict): Additional keyword arguments for element symbol text.
        annotations (dict[str, str] | Callable[[np.ndarray], str] | None): Annotation to
            display for each element tile. Can be either:
            - dict mapping element symbols to annotation strings
            - callable that takes histogram values and returns annotation string
            - None for not displaying annotations (default)

        --- Element type colors ---
        color_elem_strategy ("symbol" | "background" | "both" | "off"): Whether to
            color element symbols, tile backgrounds, or both based on element type.
            Defaults to "background".
        elem_type_colors (dict | None): dict to map element types to colors.
            None to use the default = pymatviz.colors.ELEM_TYPE_COLORS.
        subplot_kwargs (dict | None): Additional keywords passed to make_subplots(). Can
            be used e.g. to toggle shared x/y-axes.
        x_axis_kwargs (dict | None): Additional keywords for x-axis like tickfont,
            showticklabels, nticks, tickformat, tickangle.
        y_axis_kwargs (dict | None): Additional keywords for y-axis.

    Returns:
        go.Figure: Plotly Figure object with histograms in a periodic table layout.
    """
    # Process data into a consistent format
    if isinstance(data, pd.DataFrame):
        data = data.to_dict("list")
    elif isinstance(data, pd.Series):
        data = data.to_dict()

    if isinstance(color_elem_strategy, dict):
        elem_type_colors = color_elem_strategy
    elif color_elem_strategy in VALID_COLOR_ELEM_STRATEGIES:
        elem_type_colors = ELEM_TYPE_COLORS
    else:
        raise ValueError(
            f"{color_elem_strategy=} must be one of {VALID_COLOR_ELEM_STRATEGIES}"
        )

    # Initialize figure with subplots in periodic table layout
    n_rows, n_cols = 10, 18
    subplot_defaults = dict(
        vertical_spacing=0.25 / n_rows,
        horizontal_spacing=0.25 / n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
    )
    fig = make_subplots(
        rows=n_rows, cols=n_cols, **subplot_defaults | (subplot_kwargs or {})
    )

    # Get all-elements x_range if not provided
    if x_range is None:
        all_values = [val for vals in data.values() for val in vals if not pd.isna(val)]
        bins_range = (min(all_values), max(all_values)) if all_values else (0, 1)
    else:
        bins_range = x_range

    # Check if row 7 is empty. If so, pull all rows below it up by one.
    row_7_is_empty = not bool(
        set(data) & {el.symbol for el in Element if el.Z in (87, 88, *range(104, 119))}
    )

    # Create histograms for each element
    for symbol, period, group, *_ in df_ptable.itertuples():
        row, col = period - 1, group - 1  # row, col are 0-indexed

        if hide_f_block in ("auto", True) and row in (6, 7) and 3 <= col <= 17:
            continue

        if row_7_is_empty and row >= 6:
            row -= 1

        subplot_idx = row * n_cols + col + 1
        subplot_key = subplot_idx if subplot_idx != 1 else ""
        xy_ref = dict(xref=f"x{subplot_key} domain", yref=f"y{subplot_key} domain")

        # Add element type background
        elem_type = df_ptable.loc[symbol].get("type", None)
        if elem_type in elem_type_colors and color_elem_strategy in {
            "background",
            "both",
        }:
            rect_pos = dict(x0=0, y0=0, x1=1, y1=1, row=row + 1, col=col + 1)
            fig.add_shape(
                type="rect",
                **rect_pos,
                fillcolor=elem_type_colors[elem_type],
                line_width=0,
                layer="below",
                **xy_ref,
                opacity=0.05,
            )

        # Skip if no data for this element
        if data.get(symbol) is None:
            continue

        values = np.asarray(data[symbol])
        values = values[~np.isnan(values)]

        if element_symbol_map is not None:
            display_symbol = element_symbol_map.get(symbol, symbol)
        else:
            display_symbol = symbol

        hover_template = (
            f"<b>{display_symbol}</b>"
            if display_symbol == symbol
            else f"<b>{display_symbol}</b> ({symbol})"
        ) + "<br>Range: %{x}<br>Count: %{y}<extra></extra>"

        start, end = 0, 1
        if bins_range and len(bins_range) == 2:
            start, end = bins_range[0] or 0, bins_range[1] or 1

        fig.add_histogram(
            x=values,
            xbins=dict(start=start, end=end, size=(end - start) / bins),
            marker_color=px.colors.sample_colorscale(colorscale, bins),
            showlegend=False,
            hovertemplate=hover_template,
            row=row + 1,
            col=col + 1,
        )

        # Add element symbol
        font_color = "lightgray"
        symbol_style = {
            "font_size": (font_size or 10) * scale,
            "font_weight": "bold",
            "xanchor": "left",
            "yanchor": "top",
            "font_color": elem_type_colors.get(elem_type, font_color)
            if color_elem_strategy in {"symbol", "both"}
            else font_color,
            "x": 0,
            "y": 1,
        } | (symbol_kwargs or {})

        fig.add_annotation(
            text=display_symbol,
            **xy_ref,
            showarrow=False,
            **symbol_style,
        )

        if annotations is not None:
            if callable(annotations):
                # Pass the element's values to the callable
                annotation = annotations(values)
            elif isinstance(annotations, dict):
                # Use dictionary lookup
                annotation = annotations.get(symbol, "")
            else:
                annotation = ""

            if annotation:  # Only add annotation if we have text
                # Convert single annotation to list for uniform handling
                for anno in (
                    [annotation] if isinstance(annotation, str | dict) else annotation
                ):
                    # Convert string annotations to dict format
                    anno_dict = {"text": anno} if isinstance(anno, str) else anno
                    anno_defaults = {
                        "font_size": (font_size or 8) * scale,
                        "x": 0.95,
                        "y": 0.95,
                        "showarrow": False,
                        "xanchor": "right",
                        "yanchor": "top",
                    }
                    fig.add_annotation(**anno_defaults | xy_ref | anno_dict)

    if colorbar is not False:
        cbar_settings = _get_colorbar_settings(
            colorbar, font_size=font_size, scale=scale
        )
        cmin, cmax = 0, 1
        if bins_range and len(bins_range) == 2:
            cmin, cmax = bins_range[0] or 0, bins_range[1] or 1
        _add_colorbar_trace(
            fig,
            colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=cbar_settings,
            row=n_rows,
            col=n_cols,
        )

    # Update global figure layout
    fig.layout.margin = dict(l=10, r=10, t=10, b=10)
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.width = 900 * scale
    fig.layout.height = 500 * scale

    # get current plotly template line colors
    import plotly.io as pio

    template_line_color = pio.templates[pio.templates.default].layout.xaxis.linecolor

    y_axis_kwargs = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        ticks="",
        showline=False,  # remove axis lines
        type="log" if log else "linear",
        linecolor=template_line_color,
    ) | (y_axis_kwargs or {})
    fig.update_yaxes(**y_axis_kwargs)
    x_axis_kwargs = dict(
        showgrid=False,  # hide grid lines
        showline=True,  # show axis line
        linecolor=template_line_color,
        zeroline=False,
        ticklen=4,
        ticks="inside",  # move ticks to inside
        mirror=False,  # only show edge line
        tickwidth=1,
        tickcolor=template_line_color,
        tickmode="auto",
        showticklabels=True,  # show x tick labels on all subplots
        nticks=3,
        tickformat=".2g",
        tickfont=dict(size=9 * scale),
    ) | (x_axis_kwargs or {})
    fig.update_xaxes(**x_axis_kwargs)

    if colorbar is not False:
        cbar_settings = _get_colorbar_settings(
            colorbar or {}, font_size=font_size, scale=scale
        )

        for trace in fig.data:
            if hasattr(trace, "marker") and hasattr(trace.marker, "colorbar"):
                trace.marker.colorbar = cbar_settings

        if log:
            fig.update_xaxes(tickformat=".4s")

    return fig


def _add_colorbar_trace(
    fig: go.Figure,
    colorscale: ColorScale,
    cmin: float,
    cmax: float,
    colorbar: dict[str, Any],
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add an invisible scatter trace with a colorbar to the figure."""
    # For callable colorscales, sample at endpoints to create a 2-point colorscale
    if callable(colorscale):
        colorscale = [[0, colorscale("", cmin, 0)], [1, colorscale("", cmax, 0)]]  # type: ignore[assignment]

    # Ensure tickformat is included in colorbar settings
    if "tickformat" not in colorbar:
        colorbar["tickformat"] = ".4s"  # SI suffix (k=1000, M=1e6, G=1e9, etc.)

    marker = dict[str, Any](
        size=0,
        color=[cmin, cmax],
        colorscale=colorscale,
        showscale=True,
        cmin=cmin,
        cmax=cmax,
        colorbar=colorbar,
    )
    scatter_kwargs = dict[str, Any](
        x=[None],
        y=[None],
        mode="markers",
        marker=marker,
        showlegend=False,
        hoverinfo="none",
    )
    if row is not None and col is not None:
        scatter_kwargs.update(row=row, col=col)
        # Hide the axes for the invisible scatter trace
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
    fig.add_scatter(**scatter_kwargs)


def _get_colorbar_settings(
    colorbar: dict[str, Any] | None,
    *,
    split_idx: int = 0,
    n_splits: int = 1,
    split_name: str | None = None,
    font_size: int | None = None,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Get colorbar settings with consistent positioning logic."""
    cbar_settings = dict(
        lenmode="fraction",
        thickness=15,
        title_font_size=scale * 1.2 * (font_size or 12),
        tickformat="~s",  # SI suffix (k=1000, M=1e6, G=1e9, etc.)
    ) | (colorbar or {})

    # Only apply orientation-specific defaults for values not set by user
    horizontal = cbar_settings.get("orientation") == "h"
    if horizontal:
        h_defaults = dict(len=0.4, x=0.4, y=0.76 - split_idx * 0.1, title_side="top")
        cbar_settings = h_defaults | cbar_settings
    else:
        bar_len = 0.87 / max(1, n_splits)
        v_defaults = dict(
            len=bar_len, x=1.02 + split_idx * 0.15, y=0.5, title_side="right"
        )
        cbar_settings = v_defaults | cbar_settings

    # Handle title
    if title := cbar_settings.get("title"):
        title_text = title.get("text", "") if isinstance(title, dict) else title
        if split_name:
            title_text = title_text or split_name
        cbar_settings["title"] = dict(text=title_text)
    elif split_name:
        cbar_settings["title"] = dict(text=split_name)

    return cbar_settings


def ptable_heatmap_splits_plotly(
    data: pd.DataFrame | pd.Series | dict[str, list[float]],
    *,
    # Split
    orientation: Literal["diagonal", "horizontal", "vertical", "grid"] = "diagonal",
    # Figure
    colorscale: ColorScale | Sequence[ColorScale] = "Viridis",
    colorbar: dict[str, Any] | Sequence[dict[str, Any]] | Literal[False] | None = None,
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | Literal["auto"] = "auto",
    # Layout
    font_size: int | None = None,
    scale: float = 1.0,
    # Symbol
    element_symbol_map: dict[str, str] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
    # Annotation
    annotations: dict[str, str | dict[str, Any]]
    | Callable[[np.ndarray], str | dict[str, Any]]
    | None = None,
    # Hover tooltip
    hover_template: str | None = None,
    hover_fmt: str | Callable[[float], str] = ".3g",
    split_value_fmt: str | Sequence[str] | None = ".3g",
    # Additional options
    nan_color: str = "#eff",
    zero_color: str = "#aaa",
    hover_data: dict[str, str | int | float] | pd.Series | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a Plotly figure with an interactive heatmap of the periodic table,
    where each element tile is split into sections representing different values.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[list[float]]]): Map from element
            symbols to plot data. E.g. if dict, {"Fe": [1, 2], "Co": [3, 4]}, where the
            1st value would be plotted in lower-left corner, 2nd in the upper-right.

        --- Figure ---
        colorscale (ColorScale | Sequence[ColorScale]): Color scale(s) for heatmap.
            If a single colorscale is provided, it will be used for all splits.
            If a sequence is provided, each colorscale will be used for its
            corresponding split. Can be:
            - str: Name of built-in colorscale ("turbo", "inferno", "plasma", ...)
            - list[str]: List of colors to interpolate between
            - list[tuple[float, str]]: List of (position, color) pairs
            - Callable[[str, float, int], str]: Function mapping (element symbol, split
              value, split index) to color string. Useful for custom color schemes.
        colorbar (dict[str, Any] | Sequence[dict[str, Any]] | None): Plotly colorbar
            properties. If a single dict is provided, it will be used as a template for
            all colorbars. If a sequence is provided, each dict will be used for its
            corresponding split. Set to False to hide all colorbars. Defaults to
            dict(orientation="v"). Each colorbar dict can include any subset of plotly
            colorbar properties, see https://plotly.com/python/reference#heatmap-colorbar.
        on_empty ("hide" | "show"): Whether to show tiles for elements without data.
            Defaults to "hide".
        hide_f_block (bool | "auto"): Hide f-block (lanthanide and actinide series).
            Defaults to "auto", meaning hide if no data present. This keyword is
            slightly imprecisely named since it will hide the whole lanthanide and
            actinide rows even though the last items Lu + Lr in those rows are d-block.
        orientation (str): How to split each element tile. Defaults to "diagonal".
            - "diagonal": Split at 45° angles
            - "horizontal": Split into equal horizontal strips
            - "vertical": Split into equal vertical strips
            - "grid": Split into 2x2 grid (only valid for n_splits=4)

        --- Layout ---
        font_size (int): Element symbol and annotation text size. Defaults to automatic
            font size based on plot size.
        scale (float): Scaling factor for whole figure layout. Defaults to 1.

        --- Symbol ---
        element_symbol_map (dict[str, str] | None): A dictionary to map element symbols
            to custom strings. If provided, these custom strings will be displayed
            instead of the standard element symbols. Defaults to None.
        symbol_kwargs (dict): Additional keyword arguments for element symbol text.
            Supports all plotly annotation properties plus font customization through
            the "font" key. Font properties include:
            - color (str): Text color. Defaults to "black".
            - size (int): Font size in pixels. Defaults to font_size * 1.5.
            - family (str): Font family. Defaults to "Arial, sans-serif".
            - weight (str): Font weight. Defaults to "bold".
            Example: dict(font=dict(color="red", size=16, weight="normal"))
            Other common kwargs:
            - x, y (float): Position within tile. Defaults to (0.5, 0.5).
            - xanchor, yanchor (str): Text anchor point. Defaults to "center", "middle".

        --- Annotation ---
        annotations (dict[str, str] | Callable[[np.ndarray], str] | None): Annotation to
            display for each element tile. Can be either:
            - dict mapping element symbols to annotation strings
            - callable that takes values and returns annotation string
            - None for not displaying annotations (default)

        --- Hover tooltip ---
        hover_template (str | None): Custom template for hover tooltip. Use {name} for
            element name, {symbol} for element symbol, {split_name} for split name,
            {value} for split value, and {hover_data} for custom hover data.
            Defaults to None, which shows element name, symbol and all split values.
        hover_fmt (str | Callable[[float], str]): Float format for values in hover
            tooltip. Can be either a format string (e.g. ".3g", ".2f") or a callable
            that takes a float value and returns a formatted string. Defaults to ".3g".
        split_value_fmt (str | Sequence[str] | Literal[False] | None): Float format for
            split values in the heatmap. Can be:
            - str: Format string for all splits
            - Sequence[str]: Format strings for each split
            - None: Hide split values, i.e. no text labels on each tile split


        --- Additional options ---
        nan_color (str): Color for NaN values. Defaults to "#eff".
        zero_color (str): Color for zero values. Defaults to "#aaa".
        hover_data (dict[str, str] | pd.Series): Additional text to append to hover
            tooltip for each element. Defaults to None.
        subplot_kwargs (dict): Additional keyword arguments for subplots.
            See https://plotly.com/python/subplots/ for more information.

        For other args, see ptable_heatmap_plotly().

    Returns:
        go.Figure: Plotly Figure object with the periodic table heatmap splits.

    Raises:
        ValueError: If n_splits not in {2, 3, 4} or orientation="grid" with n_splits!=4
    """
    import plotly.colors
    from plotly.validator_cache import ValidatorCache
    from pymatgen.core import Element

    colorscale_validator = ValidatorCache.get_validator("scatter.marker", "colorscale")

    # Get split names if data is a DataFrame
    split_labels: list[str] = []
    if isinstance(data, pd.DataFrame):
        split_labels = list(data.columns)
        # Propagate column names to colorbar titles if not explicitly set
        if isinstance(colorbar, dict):
            colorbar = colorbar.copy()
            if "title" not in colorbar:
                colorbar["title"] = split_labels[0] if len(split_labels) == 1 else None
        elif isinstance(colorbar, Sequence):
            colorbar = list(colorbar)  # Convert to list to allow modification
            for idx, (cbar, label) in enumerate(
                zip(colorbar, split_labels, strict=False)
            ):
                if isinstance(cbar, dict) and "title" not in cbar:
                    colorbar[idx] = cbar.copy()
                    colorbar[idx]["title"] = label
        elif colorbar is None:
            # Create colorbar settings with column names as titles
            colorbar = [dict(title=label) for label in split_labels]
        data = {idx: row.tolist() for idx, row in data.iterrows()}
    elif isinstance(data, pd.Series):
        data = data.to_dict()

    # Calculate split ranges early if using multiple colorscales
    n_splits = len(next(iter(data.values())))

    if not split_labels:  # if not DataFrame or empty columns
        split_labels = [f"Split {idx + 1}" for idx in range(n_splits)]

    use_multiple_cbar = (
        isinstance(colorscale, Sequence) and not isinstance(colorscale, str)
    ) or isinstance(colorbar, Sequence)

    # Calculate ranges per split
    split_ranges = []
    for split_idx in range(n_splits):
        split_values = [
            values[split_idx]
            for values in data.values()
            if len(values) > split_idx
            and not np.isnan(values[split_idx])
            and values[split_idx] != 0
        ]
        if split_values:
            split_ranges.append((min(split_values), max(split_values)))
        else:
            split_ranges.append((0, 1))  # Default range if no valid values

    if use_multiple_cbar:
        # Multiple colorscales/colorbars mode
        colorscales = (
            [colorscale] * n_splits
            if not isinstance(colorscale, Sequence) or isinstance(colorscale, str)
            else colorscale
        )
        colorbars = (
            [colorbar or {}] * len(colorscales)
            if not isinstance(colorbar, Sequence)
            else colorbar
        )

        # Validate lengths
        if len(colorscales) != n_splits:
            raise ValueError(
                f"Number of colorscales ({len(colorscales)}) must match {n_splits=}"
            )
        if len(colorbars) != n_splits:
            raise ValueError(
                f"Number of colorbars ({len(colorbars)}) must match {n_splits=}"
            )
    else:
        # Single colorscale/colorbar mode (default)
        colorscales = [colorscale] * n_splits  # type: ignore[assignment]
        colorbars = [colorbar or {}]  # type: ignore[list-item]

    # Validate colorscales
    validator = colorscale_validator
    for idx, cscale in enumerate(colorscales):
        if callable(cscale):
            continue
        if isinstance(cscale, str):
            colorscales[idx] = validator.validate_coerce(cscale)  # type: ignore[index]
        else:
            colorscales[idx] = validator.validate_coerce(list(cscale))  # type: ignore[index]

    # Initialize figure with subplots
    has_f_block_data = any(
        Element(symbol).is_actinoid or Element(symbol).is_lanthanoid for symbol in data
    )
    should_hide_f_block = hide_f_block is True or (
        not has_f_block_data and hide_f_block == "auto"
    )
    n_rows, n_cols = 7 if should_hide_f_block else 10, 18
    subplot_kwargs = dict(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=0.01,
        horizontal_spacing=0.001,
    ) | (subplot_kwargs or {})
    fig = make_subplots(**subplot_kwargs)

    # warn about unrecognized element symbols
    unrecognized_element_symbols = set(data) - {*map(str, Element)}
    if unrecognized_element_symbols:
        warnings.warn(
            f"{unrecognized_element_symbols=}\nShould be simple strings of element "
            "symbols",
            stacklevel=2,
        )

    def create_section_coords(
        n_splits: Literal[2, 3, 4],
        orientation: Literal["diagonal", "horizontal", "vertical", "grid"],
    ) -> list[tuple[list[float], list[float]]]:
        """Generate x,y coordinates to split a unit square into n equal sections."""
        if orientation == "grid":
            if n_splits != 4:
                raise ValueError(
                    f"{orientation=} is only supported for n_splits=4, got {n_splits=}"
                )
            return [  # Split into 2x2 grid of squares
                ([0, 0.5, 0.5, 0], [0, 0, 0.5, 0.5]),  # top-left
                ([0.5, 1, 1, 0.5], [0, 0, 0.5, 0.5]),  # top-right
                ([0, 0.5, 0.5, 0], [0.5, 0.5, 1, 1]),  # bottom-left
                ([0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1]),  # bottom-right
            ]

        if orientation == "horizontal":
            # Split into equal horizontal strips
            height = 1 / n_splits
            return [
                (
                    [0.0, 1.0, 1.0, 0.0],  # x-coordinates
                    [ii * height, ii * height, (ii + 1) * height, (ii + 1) * height],
                )
                for ii in range(n_splits)
            ][::-1]  # reverse to maintain top-to-bottom order

        if orientation == "vertical":
            # Split into equal vertical strips
            width = 1 / n_splits
            return [
                (
                    [ii * width, (ii + 1) * width, (ii + 1) * width, ii * width],
                    [0.0, 0.0, 1.0, 1.0],  # y-coordinates
                )
                for ii in range(n_splits)
            ]

        # orientation == "diagonal"
        if n_splits == 2:
            return [
                ([0, 0, 1, 0], [0, 1, 1, 0]),  # top-left triangle
                ([0, 1, 1, 0], [0, 0, 1, 0]),  # bottom-right triangle
            ]
        mid = 0.5
        if n_splits == 3:
            return [  # upside-down Y-shaped split
                ([0, 1, 1, mid, 0, 0], [0, 0, 0.3, mid, 0.3, 0]),  # bottom
                ([0, mid, mid, 0, 0, 0], [1, 1, mid, 0.3, 0, 1]),  # top-left
                ([1, mid, mid, 1, 1, 1], [1, 1, mid, 0.3, 1, 1]),  # top-right
            ]
        # n_splits == 4, diagonal orientation
        return [  # split square into 4 equal triangles whose tips meet at center
            ([0, 1, mid, 0], [0, 0, mid, 0]),  # bottom
            ([0, 0, mid, 0], [0, 1, mid, 0]),  # left
            ([1, 1, mid, 1], [0, 1, mid, 0]),  # right
            ([0, 1, mid, 0], [1, 1, mid, 1]),  # top
        ]

    # Check if row 7 is empty. If so, pull all rows below it up by one.
    row_7_is_empty = not bool(
        set(data) & {el.symbol for el in Element if el.Z in (87, 88, *range(104, 119))}
    )

    # Process data and create shapes for each element
    for symbol, period, group, _name, *_ in df_ptable.itertuples():
        row, col = period - 1, group - 1  # row, col are 0-indexed
        if symbol not in data and on_empty == "hide":
            continue

        # Skip f-block elements if hide_f_block is True or if it's auto and there is no
        # data to show for those elements
        elem = Element(symbol)
        if (elem.is_actinoid or elem.is_lanthanoid) and should_hide_f_block:
            continue

        if row_7_is_empty and row > 6:
            row -= 1

        subplot_idx = row * n_cols + col + 1
        subplot_key = subplot_idx if subplot_idx != 1 else ""
        xy_ref = dict(xref=f"x{subplot_key}", yref=f"y{subplot_key}")

        # Get values and colors
        values = np.asarray(data.get(symbol, []), dtype=float)

        if len(values) not in {2, 3, 4}:
            raise ValueError(f"Number of splits {len(values)} must be 2, 3, or 4")

        # Create sections
        sections = create_section_coords(len(values), orientation)  # type: ignore[arg-type]
        split_colors: list[str] = []  # Store colors for each split
        for idx, (xs, ys) in enumerate(sections):  # Loop over element tile splits
            if values[idx] == 0:
                color = zero_color
            elif len(values) <= idx or np.isnan(values[idx]):
                color = nan_color
            elif callable(colorscale):
                # Use the callable to get color directly
                color = colorscale(symbol, values[idx], idx)
            else:
                # Get the colorscale for this split
                cscale = colorscales[idx]

                # Get range for this split
                cmin, cmax = split_ranges[idx]

                # Calculate normalized position in colorscale
                scale_pos = (values[idx] - cmin) / (cmax - cmin)
                # Clamp scale_pos to [0, 1] to handle values outside the range
                scale_pos = max(0, min(1, scale_pos))

                # Use plotly builtin color interpolation logic
                if isinstance(cscale, str):
                    # For string colorscales, use plotly's get_colorscale
                    cscale = plotly.colors.get_colorscale(cscale)
                elif isinstance(cscale, list | tuple) and not isinstance(
                    cscale[0], list | tuple
                ):
                    # If colorscale is list of colors, convert to proper format
                    n_colors = len(cscale)
                    cscale = [
                        [idx / (n_colors - 1), color]
                        for idx, color in enumerate(cscale)
                    ]  # type: ignore[assignment]
                elif not isinstance(cscale, list | tuple):
                    raise ValueError(
                        f"Invalid colorscale type: {type(cscale)}. Must be string, "
                        "list of colors, or list of (position, color) pairs."
                    )
                color = plotly.colors.sample_colorscale(cscale, [scale_pos])[0]

            split_colors.append(color)

            fig.add_scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color="white", width=0),
                hoverinfo="skip",
                showlegend=False,
                row=row + 1,
                col=col + 1,
            )

            # Add formatted split value annotation
            if split_value_fmt is not None and not np.isnan(values[idx]):
                current_fmt = (
                    split_value_fmt[idx]
                    if isinstance(split_value_fmt, Sequence)
                    and not isinstance(split_value_fmt, str)
                    else split_value_fmt
                )
                anno_text = f"{values[idx]:{current_fmt}}"

                # Determine annotation position and anchors
                pos_config = {  # Default centered, should be overridden
                    "x": 0.5,
                    "y": 0.5,
                    "xanchor": "center",
                    "yanchor": "middle",
                }
                n_s = len(values)  # Use actual n_splits for the element

                if orientation == "diagonal":
                    if n_s == 2:
                        # idx map: 0 -> top-left, 1 -> bottom-right
                        pos_config = {
                            "x": (0, 1)[idx],
                            "y": (1, 0)[idx],
                            "xanchor": ("left", "right")[idx],
                            "yanchor": ("top", "bottom")[idx],
                        }
                    elif n_s == 3:
                        # idx map: 0 -> bottom, 1 -> top-left, 2 -> top-right
                        pos_config = {
                            "x": (0.5, 0.05, 0.95)[idx],
                            "y": (0.05, 0.95, 0.95)[idx],
                            "xanchor": ("center", "left", "right")[idx],
                            "yanchor": ("bottom", "top", "top")[idx],
                        }
                    elif n_s == 4:  # Order: bottom, left, right, top
                        # idx map: 0 -> bottom, 1 -> left, 2 -> right, 3 -> top
                        pos_config = {
                            "x": (0.5, 0.05, 0.95, 0.5)[idx],
                            "y": (0.05, 0.5, 0.5, 0.95)[idx],
                            "xanchor": ("center", "left", "right", "center")[idx],
                            "yanchor": ("bottom", "middle", "middle", "top")[idx],
                        }

                elif orientation == "horizontal":
                    # Sections are ordered from top to bottom in create_section_coords
                    strip_height = 1 / n_s
                    # Place at top-left of each strip
                    pos_config = {
                        "x": 0.05,
                        "y": (1 - idx * strip_height) - 0.05,
                        "xanchor": "left",
                        "yanchor": "top",
                    }

                elif orientation == "vertical":
                    # Sections are ordered from left to right
                    strip_width = 1 / n_s
                    # Place at top-left of each strip
                    pos_config = {
                        "x": (idx * strip_width) + 0.05,
                        "y": 0.95,
                        "xanchor": "left",
                        "yanchor": "top",
                    }

                elif orientation == "grid":  # n_splits must be 4
                    if idx == 0:  # Top-left
                        pos_config = {
                            "x": 0.05,
                            "y": 0.95,
                            "xanchor": "left",
                            "yanchor": "top",
                        }
                    elif idx == 1:  # Top-right
                        pos_config = {
                            "x": 0.95,
                            "y": 0.95,
                            "xanchor": "right",
                            "yanchor": "top",
                        }
                    elif idx == 2:  # Bottom-left
                        pos_config = {
                            "x": 0.05,
                            "y": 0.05,
                            "xanchor": "left",
                            "yanchor": "bottom",
                        }
                    elif idx == 3:  # Bottom-right
                        pos_config = {
                            "x": 0.95,
                            "y": 0.05,
                            "xanchor": "right",
                            "yanchor": "bottom",
                        }

                section_luminance = luminance(color)
                anno_font_color = "black" if section_luminance > 0.55 else "white"
                anno_font_size = (font_size or 8) * scale

                fig.add_annotation(
                    text=anno_text,
                    font=dict(color=anno_font_color, size=anno_font_size),
                    showarrow=False,
                    **pos_config,  # Contains x, y, xanchor, yanchor
                    **xy_ref,
                )

        if element_symbol_map is not None:
            display_symbol = element_symbol_map.get(symbol, symbol)
        else:
            display_symbol = symbol

        # Calculate text color based on background color
        mean_luminance = sum(luminance(color) for color in split_colors) / len(
            split_colors
        )
        high_contrast_text_color = "black" if mean_luminance > 0.55 else "white"
        symbol_defaults = dict(x=0.5, y=0.5, showarrow=False)
        symbol_kwargs = symbol_kwargs or {}
        font_defaults = dict(
            size=(font_size or 14) * scale, color=high_contrast_text_color
        )
        fig.add_annotation(
            text=display_symbol,
            **symbol_defaults
            | xy_ref
            | symbol_kwargs
            | dict(font=font_defaults | symbol_kwargs.get("font", {})),
        )

        # Add hover data
        elem_name = df_ptable.loc[symbol, "name"]
        split_name: str | None  # for mypy
        if hover_template is not None:
            # Use custom hover template
            split_values = []
            for idx, val in enumerate(values):
                if not np.isnan(val):
                    split_name = split_labels[idx]
                    formatted_val = (
                        hover_fmt(val) if callable(hover_fmt) else f"{val:{hover_fmt}}"
                    )
                    split_values.append(
                        hover_template.format(
                            name=elem_name,
                            symbol=symbol,
                            split_name=split_name,
                            value=formatted_val,
                            hover_data=hover_data.get(symbol, "") if hover_data else "",
                        )
                    )
            hover_text = "<br>".join(map(str, split_values))
        else:
            # Use default hover template
            hover_text = f"<b>{elem_name}</b>"
            for idx, val in enumerate(values):
                if not np.isnan(val):
                    split_name = split_labels[idx]
                    formatted_val = (
                        hover_fmt(val) if callable(hover_fmt) else f"{val:{hover_fmt}}"
                    )
                    hover_text += f"<br>{split_name}: {formatted_val}"

            if hover_data and symbol in hover_data:
                hover_text += f"<br>{hover_data[symbol]}"

        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=hover_text,
            showarrow=False,
            opacity=0,
            hovertext=hover_text,
            **xy_ref,
        )
        if annotations is not None:
            if callable(annotations):
                # Pass the element's values to the callable
                annotation = annotations(values)
            elif isinstance(annotations, dict):
                # Use dictionary lookup
                annotation = annotations.get(symbol, "")
            else:
                annotation = ""

            if annotation:  # Only add annotation if we have text
                # Convert single annotation to list for uniform handling
                for anno in (
                    [annotation] if isinstance(annotation, str | dict) else annotation
                ):
                    # Convert string annotations to dict format
                    anno_dict = {"text": anno} if isinstance(anno, str) else anno
                    anno_defaults = {
                        "font_size": (font_size or 8) * scale,
                        "x": 0.95,
                        "y": 0.95,
                        "showarrow": False,
                        "xanchor": "right",
                        "yanchor": "top",
                        "font": dict(
                            color=high_contrast_text_color,
                            size=(font_size or 8) * scale,
                        ),
                    }
                    fig.add_annotation(**anno_defaults | xy_ref | anno_dict)

    # Update layout
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.width = 45 * n_cols * scale
    fig.layout.height = 45 * n_rows * scale + 40
    fig.layout.margin = dict(l=10, r=10, t=50, b=10)
    fig.layout.xaxis.visible = False
    fig.layout.yaxis.visible = False

    # Update subplot x and y axes to be invisible and fixed range
    fig.update_xaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        range=[0, 1],
        scaleanchor="y",
    )
    fig.update_yaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        range=[0, 1],
        scaleratio=1,  # ensure square tiles
    )

    # Add colorbar
    if colorbar is not False:
        split_names = {
            "diagonal": {
                2: ["bottom-left", "top-right"],
                3: ["bottom", "top-left", "top-right"],
                4: ["bottom", "left", "right", "top"],
            },
            "horizontal": {
                2: ["bottom", "top"],
                3: ["bottom", "middle", "top"],
                4: ["bottom", "lower-middle", "upper-middle", "top"],
            },
            "vertical": {
                2: ["left", "right"],
                3: ["left", "middle", "right"],
                4: ["left", "left-middle", "right-middle", "right"],
            },
            "grid": {4: ["top-left", "top-right", "bottom-left", "bottom-right"]},
        }[orientation][len(colorscales)]

        # Calculate positions for vertical and horizontal colorbars
        for split_idx, (cscale, split_name) in enumerate(
            zip(colorscales, split_names, strict=True)
        ):
            if not use_multiple_cbar and split_idx > 0:
                continue  # Skip if not using multiple colorbars and not first split

            cbar = colorbars[split_idx if use_multiple_cbar else 0]
            cmin, cmax = split_ranges[split_idx]

            # Get colorbar settings with consistent positioning
            cbar_settings = _get_colorbar_settings(
                cbar,
                split_idx=split_idx,
                n_splits=n_splits if use_multiple_cbar else 1,
                split_name=split_name if use_multiple_cbar else None,
                font_size=font_size,
                scale=scale,
            )
            _add_colorbar_trace(fig, cscale, cmin, cmax, cbar_settings)  # type: ignore[arg-type]

    # Apply SI suffixes to all colorbars
    if colorbar is not False:
        for idx, trace in enumerate(fig.data):
            if hasattr(trace, "marker") and hasattr(trace.marker, "colorbar"):
                # Get the appropriate colorbar settings for this split
                split_idx = idx % n_splits
                split_name = split_labels[split_idx] if split_labels else None
                cb = (
                    colorbars[split_idx]
                    if isinstance(colorbar, Sequence)
                    else colorbars[0]
                )
                cbar_settings = _get_colorbar_settings(
                    cb,
                    split_idx=split_idx,
                    n_splits=n_splits,
                    split_name=split_name,
                    font_size=font_size,
                    scale=scale,
                )
                trace.marker.colorbar = cbar_settings

    return fig


ElemData: TypeAlias = (
    tuple[Sequence[float], Sequence[float]]
    | tuple[Sequence[float], Sequence[float], Sequence[float | str]]
    | dict[str, tuple[Sequence[float], Sequence[float]]]
)


def ptable_scatter_plotly(
    data: Mapping[str, ElemData],
    *,
    # Plot mode
    mode: Literal["markers", "lines", "lines+markers"] = "markers",
    # Color settings
    colorscale: str = "viridis",
    colorbar: dict[str, Any] | Literal[False] | None = None,
    # Line-specific
    x_range: tuple[float | None, float | None] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
    # Layout
    font_size: int | None = None,
    scale: float = 1.0,
    hide_f_block: bool | Literal["auto"] = False,
    # Symbol
    element_symbol_map: dict[str, str] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
    # Marker and line styling
    marker_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    # Annotation
    annotations: dict[str, str | dict[str, Any]]
    | Callable[[ElemData], str | dict[str, Any] | list[dict[str, Any]]]
    | None = None,
    # Element type colors
    color_elem_strategy: ColorElemTypeStrategy = "symbol",
    elem_type_colors: dict[str, str] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    # Axis styling
    x_axis_kwargs: dict[str, Any] | None = None,
    y_axis_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a Plotly figure with scatter/line plots for each element laid out in a
    periodic table.

    Args:
        data (dict[str, ElemData]): Map from element symbols to either:
            1. (x, y) or (x, y, color) data points for a single line/scatter plot
               E.g. {"Fe": ([1, 2, 3], [4, 5, 6])} plots points (1,4), (2,5), (3,6)
            2. dict of (x, y) tuples for multiple lines per element
               E.g. {"Fe": {"line1": ([1, 2], [3, 4]), "line2": ([2, 3], [4, 5])}}
               In this case, a legend will be shown instead of a colorbar
        mode ("markers" | "lines" | "lines+markers"): Plot mode. Defaults to "markers".

        --- Color settings ---
        colorscale (str): Colorscale to use for numeric color data. Defaults to
            "viridis".
        colorbar (dict | False | None): Colorbar settings. Defaults to None.

        --- Line-specific ---
        x_range (tuple[float | None, float | None]): x-axis range for all line plots.
            Defaults to None, meaning auto-range.
        y_range (tuple[float | None, float | None]): y-axis range for all line plots.
            Defaults to None, meaning auto-range.

        --- Layout ---
        font_size (int): Element symbol and annotation text size. Defaults to automatic
            font size based on plot size.
        scale (float): Scaling factor for whole figure layout. Defaults to 1.
        hide_f_block (bool | "auto"): Whether to hide the F-block elements. If "auto",
            hide F-block elements if row 7 is empty. Defaults to False.

        --- Symbol ---
        element_symbol_map (dict[str, str] | None): A dictionary to map element symbols
            to custom strings. If provided, these custom strings will be displayed
            instead of the standard element symbols. Defaults to None.
        symbol_kwargs (dict): Additional keyword arguments for element symbol text.
            Supports all plotly annotation properties plus font customization through
            the "font" key. Font properties include:
            - color (str): Text color. Defaults to "black".
            - size (int): Font size in pixels. Defaults to font_size * 1.5.
            - family (str): Font family. Defaults to "Arial, sans-serif".
            - weight (str): Font weight. Defaults to "bold".
            Example: dict(font=dict(color="red", size=16, weight="normal"))
            Other common kwargs:
            - x, y (float): Position within tile. Defaults to (0.5, 0.5).
            - xanchor, yanchor (str): Text anchor point. Defaults to "center", "middle".

        --- Annotation ---
        annotations (dict[str, str] | Callable[[np.ndarray], str] | None): Annotation to
            display for each element tile. Can be either:
            - dict mapping element symbols to annotation strings
            - callable that takes values and returns annotation string
            - None for not displaying annotations (default)

        --- Element type colors ---
        color_elem_strategy ("symbol" | "background" | "both" | "off"): Whether to
            color element symbols, tile backgrounds, or both based on element type.
            Defaults to "background".
        elem_type_colors (dict | None): dict to map element types to colors.
            None to use the default = pymatviz.colors.ELEM_TYPE_COLORS.

        --- Subplot ---
        subplot_kwargs (dict | None): Additional keywords passed to make_subplots().

        --- Axis styling ---
        x_axis_kwargs (dict | None): Additional keywords for x-axis like tickfont,
            showticklabels, nticks, tickformat, tickangle.
        y_axis_kwargs (dict | None): Additional keywords for y-axis.

        --- Line/marker styling ---
        line_kwargs (dict | None): Additional keywords for line plots like color,
            width, dash.
        marker_kwargs (dict | None): Additional keywords for marker plots like color,
            size, symbol.

    Returns:
        go.Figure: Plotly Figure object with line plots in a periodic table layout.
    """
    if isinstance(color_elem_strategy, dict):
        elem_type_colors = color_elem_strategy
    elif color_elem_strategy in VALID_COLOR_ELEM_STRATEGIES:
        elem_type_colors = ELEM_TYPE_COLORS
    else:
        raise ValueError(
            f"{color_elem_strategy=} must be one of {VALID_COLOR_ELEM_STRATEGIES}"
        )

    # Initialize figure with subplots
    n_rows, n_cols = 10, 18
    subplot_kwargs = dict(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=0.035,
        horizontal_spacing=0.01,
    ) | (subplot_kwargs or {})
    fig = make_subplots(**subplot_kwargs)

    # get current plotly template line colors
    import plotly.io as pio

    template_line_color = pio.templates[pio.templates.default].layout.xaxis.linecolor

    # Get global x and y ranges if not provided
    if x_range is None or y_range is None:
        all_x_vals: list[float] = []
        all_y_vals: list[float] = []
        # _* to ignore optional color data
        for elem_data in data.values():
            for x_vals, y_vals, *_ in (
                # handle both single line and multiple lines per element cases
                elem_data if isinstance(elem_data, dict) else {"": elem_data}  # type: ignore[dict-item]
            ).values():
                all_x_vals.extend(x_vals)
                all_y_vals.extend(y_vals)

        if x_range is None:
            x_range = (min(all_x_vals), max(all_x_vals))
        if y_range is None:
            y_range = (min(all_y_vals), max(all_y_vals))

    # Get default marker and line settings
    marker_defaults = dict(
        size=6 * scale, color=template_line_color, line=dict(width=0)
    )
    line_defaults = dict(width=1 * scale, color=template_line_color)

    # Find global color range if any numeric color values exist
    cbar_min, cbar_max = float("inf"), float("-inf")
    for elem_values in data.values():
        for elem_data in (
            elem_values if isinstance(elem_values, dict) else {"": elem_values}  # type: ignore[dict-item]
        ).values():
            if len(elem_data) > 2:  # Has color data
                color_vals = elem_data[2]
                if all(isinstance(val, int | float) for val in color_vals):
                    cbar_min = min(cbar_min, *color_vals)
                    cbar_max = max(cbar_max, *color_vals)

    has_numeric_colors = cbar_min != float("inf")

    # Check if row 7 is empty. If so, pull all rows below it up by one.
    row_7_is_empty = not bool(
        set(data) & {el.symbol for el in Element if el.Z in (87, 88, *range(104, 119))}
    )

    # Track whether we're plotting multiple lines per element
    line_colors: dict[str, str] | None = None
    has_multiple_lines = any(isinstance(val, dict) for val in data.values())

    if has_multiple_lines:
        # Get all unique line names across elements for consistent colors
        line_names = {
            key
            for elem_data in data.values()
            if isinstance(elem_data, dict)
            for key in elem_data
        }
        # Create color map for line names
        import plotly.express as px

        color_sequence = px.colors.qualitative.Set2
        line_colors = {
            name: color_sequence[idx % len(color_sequence)]
            for idx, name in enumerate(sorted(line_names))
        }

    for symbol, period, group, elem_name, *_ in df_ptable.itertuples():
        if symbol not in data:
            continue

        row, col = period - 1, group - 1  # row, col are 0-indexed

        if hide_f_block in ("auto", True) and row_7_is_empty and row >= 6:
            continue

        subplot_idx = row * n_cols + col + 1
        subplot_key = subplot_idx if subplot_idx != 1 else ""
        xy_ref = dict(xref=f"x{subplot_key} domain", yref=f"y{subplot_key} domain")

        # Add element type background
        elem_type = df_ptable.loc[symbol].get("type", None)
        if elem_type in elem_type_colors and color_elem_strategy in {
            "background",
            "both",
        }:
            rect_pos = dict(x0=0, y0=0, x1=1, y1=1, row=row + 1, col=col + 1)
            fig.add_shape(
                type="rect",
                **rect_pos,
                fillcolor=elem_type_colors[elem_type],
                line_width=0,
                layer="below",
                **xy_ref,
                opacity=0.05,
            )

        # Add line plot if data exists for this element
        elem_data = (
            data[symbol] if isinstance(data[symbol], dict) else {"": data[symbol]}  # type: ignore[dict-item]
        )
        for line_name, elem_vals in elem_data.items():  # type: ignore[union-attr]
            x_vals, y_vals, *rest = elem_vals
            color_vals = rest[0] if rest else None  # if 3-tuple, first entry is color

            line_props = line_defaults.copy()
            marker_props = marker_defaults.copy()

            # 1. Start with default colors
            if "markers" in mode:
                marker_props["color"] = template_line_color
            if "lines" in mode:
                line_props["color"] = template_line_color

            # 2. Apply element type colors if enabled
            if color_elem_strategy in {"symbol", "both"}:
                elem_color = elem_type_colors.get(elem_type, template_line_color)
                if "markers" in mode:
                    marker_props["color"] = elem_color
                if "lines" in mode:
                    line_props["color"] = elem_color

            # 3. Apply user settings
            if line_kwargs:
                line_props.update(line_kwargs)
            if marker_kwargs:
                marker_props.update(marker_kwargs)

            # 4. Override with line colors for multi-line plots
            if has_multiple_lines and line_name and line_colors:
                if line_name not in line_colors:
                    raise ValueError(f"{line_name=} not found in {line_colors=}")
                line_color = line_colors[line_name]
                if "markers" in mode:
                    marker_props["color"] = line_color
                if "lines" in mode:
                    line_props["color"] = line_color

            # 5. Finally, override with color data if provided (highest precedence)
            if color_vals is not None:
                if all(isinstance(v, int | float) for v in color_vals):
                    # For numeric colors, use the colorscale
                    marker_props.update(
                        color=color_vals,
                        colorscale=colorscale,
                        cmin=cbar_min,
                        cmax=cbar_max,
                        showscale=False,  # Don't show individual colorbars
                    )
                else:
                    # For discrete colors (strings), use as-is
                    marker_props["color"] = color_vals
                line_props["color"] = None

            scatter_kwargs = dict(
                x=x_vals,
                y=y_vals,
                mode=mode,
                name=line_name or None,  # None for empty string to avoid legend entry
                showlegend=bool(line_name) and symbol == next(iter(data)),
                legendgroup=line_name,
                row=row + 1,
                col=col + 1,
                line=line_props,
                marker=marker_props,
                hovertemplate=(
                    f"{elem_name}{f' - {line_name}' if line_name else ''}<br>"
                    f"x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>"
                ),
            )
            fig.add_scatter(**scatter_kwargs)

        # Add element symbol
        if element_symbol_map is not None:
            display_symbol = element_symbol_map.get(symbol, symbol)
        else:
            display_symbol = symbol

        symbol_defaults = dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(
                color=elem_type_colors.get(elem_type, template_line_color)
                if color_elem_strategy in {"symbol", "both"}
                else template_line_color,
                size=(font_size or 11) * scale,
            ),
        )
        fig.add_annotation(
            text=display_symbol, **symbol_defaults | xy_ref | (symbol_kwargs or {})
        )

        # Add custom annotations if provided
        if annotations is not None:
            if callable(annotations):
                # Pass the element's values to the callable
                annotation = annotations(data[symbol])
            elif isinstance(annotations, dict):
                annotation = annotations.get(symbol, "")
            else:
                annotation = ""

            if annotation:  # Only add annotation if we have text
                # Convert single annotation to list for uniform handling
                for anno in (
                    [annotation] if isinstance(annotation, str | dict) else annotation
                ):
                    # Convert string annotations to dict format
                    anno_dict = {"text": anno} if isinstance(anno, str) else anno
                    anno_defaults = {
                        "font_size": (font_size or 8) * scale,
                        "x": 0.95,
                        "y": 0.95,
                        "showarrow": False,
                        "xanchor": "right",
                        "yanchor": "top",
                    }
                    fig.add_annotation(**anno_defaults | xy_ref | anno_dict)

    # Update layout
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.width = 850 * scale
    fig.layout.height = 500 * scale
    fig.layout.margin = dict(l=10, r=10, t=50, b=10)

    # Update x and y axes
    axis_defaults = dict(
        showgrid=False,  # hide grid lines
        showline=True,  # show axis line
        linecolor=template_line_color,
        linewidth=1,
        mirror=False,  # only show edge line
        ticks="inside",  # move ticks to inside
        tickwidth=1,
        tickcolor=template_line_color,
        # Configure tick count
        nticks=2,  # show only 2 ticks by default
        tickmode="auto",  # let plotly choose nice tick values
        zeroline=False,  # remove x/y=0 line
        tickfont=dict(size=9 * scale),
    )

    x_axis_defaults = axis_defaults | dict(range=x_range)
    y_axis_defaults = axis_defaults | dict(range=y_range)

    fig.update_xaxes(**x_axis_defaults | (x_axis_kwargs or {}))
    fig.update_yaxes(**y_axis_defaults | (y_axis_kwargs or {}))

    # Add colorbar or legend based on plot type
    if has_multiple_lines:
        # Configure legend instead of colorbar
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.74,
                xanchor="center",
                x=0.4,
                font=dict(size=(font_size or 10) * scale),
            ),
        )

    if has_numeric_colors and colorbar is not False:
        colorbar = dict(orientation="h", lenmode="fraction", thickness=15) | (
            colorbar or {}
        )
        horizontal_cbar = colorbar.get("orientation") == "h"
        if horizontal_cbar:
            h_defaults = dict(
                x=0.4,
                y=0.74,
                title_side="top",
                len=0.4,
                title_font_size=scale * 1.2 * (font_size or 12),
            )
            colorbar = h_defaults | colorbar
        else:  # make title vertical
            v_defaults = dict(title_side="right", len=0.87)
            colorbar = v_defaults | colorbar

        if title := colorbar.get("title"):
            # <br> to increase title standoff
            colorbar["title"] = f"{title}" if horizontal_cbar else f"<br><br>{title}"

        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                showscale=True,
                cmin=cbar_min,
                cmax=cbar_max,
                colorbar=colorbar,
            ),
            hoverinfo="none",
            showlegend=False,
        )

    # Apply colorbar settings
    if colorbar is not False and has_numeric_colors:
        cbar_settings = _get_colorbar_settings(
            colorbar or {}, font_size=font_size, scale=scale
        )

        for trace in fig.data:
            if hasattr(trace, "marker") and hasattr(trace.marker, "colorbar"):
                trace.marker.colorbar = cbar_settings

    return fig
