"""Periodic table plots powered by plotly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pymatviz.colors import ELEM_TYPE_COLORS
from pymatviz.enums import ElemCountMode
from pymatviz.process_data import count_elements
from pymatviz.utils import (
    VALID_COLOR_ELEM_STRATEGIES,
    ColorElemTypeStrategy,
    ElemValues,
    df_ptable,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal


def ptable_heatmap_plotly(
    values: ElemValues,
    *,
    count_mode: ElemCountMode = ElemCountMode.composition,
    colorscale: str | Sequence[str] | Sequence[tuple[float, str]] = "viridis",
    show_scale: bool = True,
    show_values: bool = True,
    heat_mode: Literal["value", "fraction", "percent"] = "value",
    fmt: str | None = None,
    hover_props: Sequence[str] | dict[str, str] | None = None,
    hover_data: dict[str, str | int | float] | pd.Series | None = None,
    font_colors: Sequence[str] = (),
    gap: float = 5,
    font_size: int | None = None,
    bg_color: str | None = None,
    nan_color: str = "#eff",
    color_bar: dict[str, Any] | None = None,
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
        fmt (str): f-string format option for heat labels. Defaults to ".1%"
            (1 decimal place) if heat_mode="percent" else ".3g".
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
        color_bar (dict[str, Any]): Plotly colorbar properties documented at
            https://plotly.com/python/reference#heatmap-colorbar. Defaults to
            dict(orientation="h"). Commonly used keys are:
            - title: colorbar title
            - titleside: "top" | "bottom" | "right" | "left"
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

    color_bar = color_bar or {}
    color_bar.setdefault("orientation", "h")
    # if values is a series with a name, use it as the colorbar title
    if isinstance(values, pd.Series) and values.name:
        color_bar.setdefault("title", values.name)

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
                if heat_mode == "percent":
                    label = f"{heat_val:{fmt or '.1%'}}"
                else:
                    default_prec = ".1f" if heat_val < 100 else ",.0f"
                    if heat_val > 1e5:
                        default_prec = ".2g"
                    label = f"{heat_val:{fmt or default_prec}}".replace("e+0", "e")

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
        **kwargs,
    )

    # Add border heatmap
    if border is not False:
        border = border or {}
        border_color = border.pop("color", "darkgray")
        border_width = border.pop("width", 2)

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

    horizontal_cbar = color_bar.get("orientation") == "h"
    if horizontal_cbar:
        defaults = dict(
            x=0.4,
            y=0.72,
            titleside="top",
            len=0.4,
            title_font_size=scale * 1.2 * (font_size or 12),
        )
        color_bar = defaults | color_bar
    else:  # make title vertical
        defaults = dict(titleside="right", len=0.87)
        color_bar = defaults | color_bar

    if title := color_bar.get("title"):
        # <br> to increase title standoff
        color_bar["title"] = f"{title}<br>" if horizontal_cbar else f"<br><br>{title}"

    if log:
        orig_min = np.floor(min(non_nan_values))
        orig_max = np.ceil(max(non_nan_values))
        tick_values = np.logspace(orig_min, orig_max, num=10, endpoint=True)

        tick_values = [round(val, -int(np.floor(np.log10(val)))) for val in tick_values]

        color_bar = dict(
            tickvals=np.log10(tick_values),
            ticktext=[f"{v * car_multiplier:.2g}" for v in tick_values],
            **color_bar,
        )

    # suffix % to colorbar title if heat_mode is "percent"
    if heat_mode == "percent" and (cbar_title := color_bar.get("title")):
        color_bar["title"] = f"{cbar_title} (%)"

    fig.update_traces(colorbar=dict(lenmode="fraction", thickness=15, **color_bar))

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
    # Layout
    font_size: int | None = None,
    scale: float = 1.0,
    # Symbol
    element_symbol_map: dict[str, str] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
    # Annotation
    anno_text: dict[str, str] | None = None,
    anno_kwargs: dict[str, Any] | None = None,
    # Element type colors
    color_elem_strategy: ColorElemTypeStrategy = "background",
    elem_type_colors: dict[str, str] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
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

        --- Layout ---
        font_size (int): Element symbol and annotation text size. Defaults to automatic
            font size based on plot size.
        scale (float): Scaling factor for whole figure layout. Defaults to 1.

        --- Symbol ---
        element_symbol_map (dict[str, str] | None): A dictionary to map element symbols
            to custom strings. If provided, these custom strings will be displayed
            instead of the standard element symbols. Defaults to None.
        symbol_kwargs (dict): Additional keyword arguments for element symbol text.

        --- Annotation ---
        anno_text (dict[str, str]): Annotation to display for each element tile.
            Defaults to None for not displaying.
        anno_kwargs (dict): Additional keyword arguments for annotation text.

        --- Element type colors ---
        color_elem_strategy ("symbol" | "background" | "both" | "off"): Whether to
            color element symbols, tile backgrounds, or both based on element type.
            Defaults to "background".
        elem_type_colors (dict | None): dict to map element types to colors.
            None to use the default = pymatviz.colors.ELEM_TYPE_COLORS.
        subplot_kwargs (dict | None): Additional keywords passed to make_subplots(). Can
            be used e.g. to toggle shared x/y-axes.

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

    # Create histograms for each element
    for symbol, period, group, *_ in df_ptable.itertuples():
        row = period - 1
        col = group - 1

        subplot_idx = row * n_cols + col + 1
        subplot_key = subplot_idx if subplot_idx != 1 else ""
        xy_ref = dict(xref=f"x{subplot_key} domain", yref=f"y{subplot_key} domain")

        elem_type = df_ptable.loc[symbol].get("type", None)
        # Add element type background
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

        # Get display symbol and create hover template
        if element_symbol_map is not None:
            display_symbol = element_symbol_map.get(symbol, symbol)
        else:
            display_symbol = symbol

        hover_template = (
            f"<b>{display_symbol}</b>"
            if display_symbol == symbol
            else f"<b>{display_symbol}</b> ({symbol})"
        ) + "<br>Range: %{x}<br>Count: %{y}<extra></extra>"

        fig.add_histogram(
            x=values,
            xbins=dict(
                start=bins_range[0],
                end=bins_range[1],
                size=(bins_range[1] - bins_range[0]) / bins,
            ),
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

        if anno_text is not None and symbol in anno_text:
            anno_style = {
                "font_size": (font_size or 8) * scale,
                "font_color": font_color,
                "x": 1,
                "y": 0.97,
                "showarrow": False,
            } | (anno_kwargs or {})

            fig.add_annotation(text=anno_text[symbol], **xy_ref, **anno_style)

    if colorbar is not False:
        colorbar = dict(orientation="h", lenmode="fraction", thickness=15) | (
            colorbar or {}
        )

        horizontal_cbar = colorbar.get("orientation") == "h"
        if horizontal_cbar:
            defaults = dict(
                x=0.4,
                y=0.76,
                titleside="top",
                len=0.4,
                title_font_size=scale * 1.2 * (font_size or 12),
            )
            colorbar = defaults | colorbar
        else:  # make title vertical
            defaults = dict(titleside="right", len=0.87)
            colorbar = defaults | colorbar

        if title := colorbar.get("title"):
            # <br> to increase title standoff
            colorbar["title"] = (
                f"{title}<br>" if horizontal_cbar else f"<br><br>{title}"
            )

        # Create an invisible scatter trace for the colorbar
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                showscale=True,
                cmin=bins_range[0],
                cmax=bins_range[1],
                colorbar=colorbar,
            ),
            row=n_rows,
            col=n_cols,
            showlegend=False,
            hoverinfo="none",  # Disable hover tooltip
        )
        # Hide the axes for the invisible scatter trace
        fig.update_xaxes(visible=False, row=n_rows, col=n_cols)
        fig.update_yaxes(visible=False, row=n_rows, col=n_cols)

    # Update global figure layout
    fig.layout.margin = dict(l=10, r=10, t=10, b=10)
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.width = 900 * scale
    fig.layout.height = 600 * scale

    # Update x/y-axes across all subplots
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        ticks="",
        showline=False,  # remove axis lines
        type="log" if log else "linear",
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        ticks="inside",
        ticklen=4,
        tickwidth=1,
        showline=True,
        mirror=False,  # only show bottom x-axis line
        linewidth=0.5,
        linecolor="lightgray",
        # more readable tick labels
        tickangle=0,
        tickfont=dict(size=(font_size or 7) * scale),
        showticklabels=True,  # show x tick labels on all subplots
        nticks=3,
        tickformat=".2",
    )

    return fig
