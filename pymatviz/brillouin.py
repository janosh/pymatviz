"""Plot the Brillouin zone of a structure."""

import re
import warnings
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pymatviz.typing import AnyStructure


def brillouin_zone_3d(
    struct: AnyStructure | Sequence[AnyStructure],
    *,
    # Surface styling
    surface_kwargs: dict[str, Any] | None = None,
    edge_kwargs: dict[str, Any] | None = None,
    # High symmetry point styling
    point_kwargs: dict[str, Any] | Literal[False] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    # High symmetry path styling
    path_kwargs: dict[str, Any] | Literal[False] | None = None,
    # Coordinate axes styling
    axes_vectors: dict[Literal["shaft", "cone"], dict[str, Any]]
    | Literal[False]
    | None = None,
    # Grid layout
    n_cols: int = 3,
    subplot_title: Callable[[AnyStructure, Hashable], str | dict[str, Any]]
    | Literal[False]
    | None = None,
) -> go.Figure:
    """Generate a 3D plotly figure of the first Brillouin zone for given structure(s).

    Args:
        struct (AnyStructure | Sequence[AnyStructure]): Structure(s) to plot.
        # Surface styling
        surface_kwargs (dict): Styling for BZ surfaces.
        edge_kwargs (dict): Styling for BZ edges.
        # High symmetry point styling
        point_kwargs (dict | Literal[False]): Styling for high symmetry points.
        label_kwargs (dict): Styling for point labels.
        # High symmetry path styling
        path_kwargs (dict | Literal[False]): Styling for paths. Set to False to disable
            plotting paths.
        # Coordinate axes styling
        axes_vectors (dict | False): Keywords for coordinate axes vectors. Split into
            2 sub dicts axes_vectors={shaft: {...}, cone: {...}}. Use nested key
            shaft.len to control vector length. Set to False to disable axes plotting.
        # Grid layout
        n_cols (int): Number of columns for subplots. Defaults to 3.
        subplot_title (Callable[[AnyStructure, Hashable], str | dict] | False, optional):
            Function to generate subplot titles. Defaults to
            lambda struct_i, idx: f"{idx}. {struct_i.formula} (spg={spg_num})". Set to
            False to hide all subplot titles.

    Returns:
        go.Figure: A plotly figure containing the first Brillouin zone(s)
    """  # noqa: E501
    import scipy.spatial as sps
    import seekpath

    from pymatviz.process_data import normalize_structures
    from pymatviz.structure.helpers import get_subplot_title

    structures = normalize_structures(struct)

    n_structs = len(structures)
    n_cols = min(n_cols, n_structs)
    n_rows = (n_structs - 1) // n_cols + 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        # needed to avoid IndexError on fig.layout.annotations[idx - 1].update(anno)
        subplot_titles=[" " for _ in range(n_structs)],
    )

    for idx, (struct_key, structure) in enumerate(structures.items(), start=1):
        # Convert pymatgen Structure to seekpath input format
        spglib_atoms = (
            structure.lattice.matrix,  # cell
            structure.frac_coords,  # positions
            [site.specie.number for site in structure],  # atomic numbers
        )
        # Get primitive structure and symmetry info using seekpath
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="dict interface .* is deprecated", module="spglib"
            )
            seekpath_dict = seekpath.get_path(spglib_atoms)

        real_space_cell = np.array(seekpath_dict["primitive_lattice"])
        spg_num = seekpath_dict["spacegroup_number"]
        spg_symbol = seekpath_dict["spacegroup_international"]

        # Get reciprocal lattice vectors (scaled by 2π)
        k_space_cell = np.linalg.inv(real_space_cell).T * 2 * np.pi

        # Generate points for Voronoi construction
        mgrid_array = np.mgrid[-1:2, -1:2, -1:2]
        px, py, pz = np.tensordot(k_space_cell, mgrid_array, axes=(0, 0))
        points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

        voronoi = sps.Voronoi(points)  # Create Voronoi cell

        # Find central point (index 13 corresponds to [0,0,0])
        center_index = 13

        # Get BZ vertices, edges and faces
        bz_vertices = []
        bz_edges = []
        bz_faces = []

        for pid, rid in zip(voronoi.ridge_points, voronoi.ridge_vertices, strict=False):
            if pid[0] == center_index or (
                pid[1] == center_index and -1 not in rid
            ):  # Skip ridges with vertices at infinity
                ridge_vertices = voronoi.vertices[rid]
                bz_edges.append(voronoi.vertices[np.r_[rid, [rid[0]]]])
                bz_faces.append(ridge_vertices)
                bz_vertices.extend(rid)

        # Remove duplicate vertices
        bz_vertices = list(set(bz_vertices))
        vertices = voronoi.vertices[bz_vertices]

        # Plot reciprocal lattice vectors
        colors = ["red", "green", "blue"]
        labels = ["b₁", "b₂", "b₃"]

        scene = f"scene{idx}"

        # Plot reciprocal lattice vectors
        if axes_vectors is not False:
            # Validate axes_vectors dict
            if axes_vectors is not None and not all(
                key in axes_vectors for key in ("shaft", "cone")
            ):
                raise KeyError("axes_vectors must contain 'shaft' and 'cone'")

            shaft_kwargs = {} if axes_vectors is None else axes_vectors.get("shaft", {})
            cone_kwargs = {} if axes_vectors is None else axes_vectors.get("cone", {})

            for vec_idx, vec in enumerate(k_space_cell):
                start, end = np.zeros(3), vec  # Vector points
                color = shaft_kwargs.get("color", colors[vec_idx])

                fig.add_scatter3d(  # vector shaft
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode="lines",
                    line=dict(color=color, width=6) | shaft_kwargs,
                    showlegend=False,
                    hoverinfo="none",
                    scene=scene,
                )

                # Add arrow head as cone at vector end (same direction but shorter)
                arrow_dir = 0.25 * (end - start)
                # Pop parameters that we handle directly to avoid conflicts
                sizeref = cone_kwargs.pop("sizeref", 0.4)
                colorscale = cone_kwargs.pop("colorscale", [[0, color], [1, color]])
                fig.add_cone(  # arrow head
                    x=[end[0]],
                    y=[end[1]],
                    z=[end[2]],
                    u=[arrow_dir[0]],
                    v=[arrow_dir[1]],
                    w=[arrow_dir[2]],
                    anchor="cm",  # Place cone tip at vector end
                    hoverinfo="none",
                    colorscale=colorscale,
                    showscale=False,
                    sizeref=sizeref,
                    scene=scene,
                    **cone_kwargs,
                )

                fig.add_scatter3d(  # vector label at the tip
                    x=[vec[0]],
                    y=[vec[1]],
                    z=[vec[2]],
                    mode="text",
                    text=[labels[vec_idx]],
                    textposition="top center",
                    textfont=dict(size=20, color=color),
                    showlegend=False,
                    scene=scene,
                )

        # Plot BZ faces using convex hull
        unique_vertices = np.array(vertices)
        hull = sps.ConvexHull(unique_vertices)

        # Calculate BZ volume
        bz_volume = hull.volume  # in inverse cubic angstrom

        # Plot faces as a single mesh
        fig.add_mesh3d(
            x=unique_vertices[:, 0],
            y=unique_vertices[:, 1],
            z=unique_vertices[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            showscale=False,
            hovertemplate=(
                "x: %{x:.2f}<br>"
                "y: %{y:.2f}<br>"
                "z: %{z:.2f}<br>"
                f"Space group: {spg_symbol} ({spg_num})<br>"
                f"BZ volume: {bz_volume:.2f} Å⁻³"
                "<extra></extra>"
            ),
            scene=scene,
            **{"color": "lightblue", "opacity": 0.3} | (surface_kwargs or {}),
        )

        # Plot edges
        for edge_vertices in bz_edges:
            fig.add_scatter3d(
                x=edge_vertices[:, 0],
                y=edge_vertices[:, 1],
                z=edge_vertices[:, 2],
                mode="lines",
                line=dict(color="black", width=5) | (edge_kwargs or {}),
                showlegend=False,
                scene=scene,
            )

        if point_kwargs is not False:  # Add high symmetry k-points
            k_points_dict = seekpath_dict[
                "point_coords"
            ]  # This is a dict of label: coords
            k_paths = seekpath_dict["path"]

            # Convert fractional k-points to Cartesian coordinates
            cart_kpoints = {
                label: np.dot(coords, k_space_cell)
                for label, coords in k_points_dict.items()
            }

            # Plot high symmetry points
            x_coords, y_coords, z_coords, point_labels = [], [], [], []
            for label, coords in cart_kpoints.items():
                x_coords += [coords[0]]
                y_coords += [coords[1]]
                z_coords += [coords[2]]
                for text, repl in [("\\Gamma", "Γ"), ("GAMMA", "Γ"), ("DELTA", "Δ")]:
                    label = label.replace(text, repl)  # noqa: PLW2901
                # use <sub> for subscripts
                pretty_label = re.sub(r"_(\d+)", r"<sub>\1</sub>", label)
                point_labels += [pretty_label]

            fig.add_scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="markers+text",
                marker=dict(color="red", size=6) | (point_kwargs or {}),
                text=point_labels,
                textposition="top center",
                textfont=dict(size=14) | (label_kwargs or {}),
                name="High Symmetry Points",
                scene=scene,
            )

            # Add paths between high symmetry points
            if path_kwargs is not False:
                for path in k_paths:
                    for path_idx in range(len(path) - 1):
                        start_point = cart_kpoints[path[path_idx]]
                        end_point = cart_kpoints[path[path_idx + 1]]

                        fig.add_scatter3d(
                            x=[start_point[0], end_point[0]],
                            y=[start_point[1], end_point[1]],
                            z=[start_point[2], end_point[2]],
                            mode="lines",
                            line=dict(color="red", width=5, dash="dash")
                            | (path_kwargs or {}),
                            showlegend=False,
                            scene=scene,
                        )

        # Calculate the bounding box of the Brillouin zone vertices
        vertices = np.array(vertices)

        # Set subplot titles
        if subplot_title is not False:
            title_func = subplot_title or get_subplot_title
            if title_func is get_subplot_title:
                anno = title_func(structure, struct_key, idx, subplot_title)  # type: ignore[call-arg]
            else:
                anno = title_func(structure, struct_key)  # type: ignore[call-arg]
                if not isinstance(anno, (str, dict)):
                    raise TypeError("Subplot title must be a string or dict")
                if isinstance(anno, str):
                    anno = {"text": anno}
            fig.layout.annotations[idx - 1].update(anno)

        # Update 3D scene properties
        no_axes_kwargs = dict(
            showticklabels=False, showgrid=False, zeroline=False, visible=False
        )

        fig.update_scenes(
            xaxis=no_axes_kwargs,
            yaxis=no_axes_kwargs,
            zaxis=no_axes_kwargs,
            aspectmode="data",
            bgcolor="rgba(90, 90, 90, 0.01)",  # Transparent background
        )

    # Calculate subplot positions with small gap
    gap = 0.01
    for idx in range(1, n_structs + 1):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1

        x_start = (col - 1) / n_cols + gap / 2
        x_end = col / n_cols - gap / 2
        y_start = 1 - row / n_rows + gap / 2  # Invert y-axis to match row order
        y_end = 1 - (row - 1) / n_rows - gap / 2

        domain = dict(x=[x_start, x_end], y=[y_start, y_end])
        fig.update_layout({f"scene{idx}": dict(domain=domain, aspectmode="data")})

    # Update overall layout
    fig.layout.height = 400 * n_rows
    fig.layout.width = 400 * n_cols
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"  # Transparent background
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"  # Transparent background
    fig.layout.margin = dict(l=0, r=0, t=0, b=0)  # Minimize margins

    return fig
