"""Plot the Brillouin zone of a structure."""

from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from scipy.spatial import ConvexHull, Voronoi


def plot_brillouin_zone_3d(
    structure: Structure,
    fig: go.Figure | None = None,
    subplot_idx: tuple[int, int] = (1, 1),
    *,
    # Surface styling
    surface_kwargs: dict[str, Any] | None = None,
    edge_kwargs: dict[str, Any] | None = None,
    # High symmetry point styling
    point_kwargs: dict[str, Any] | Literal[False] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    # High symmetry path styling
    path_kwargs: dict[str, Any] | Literal[False] | None = None,
) -> go.Figure:
    """Generate a 3D plotly figure of the first Brillouin zone for a given structure.

    Args:
        structure (Structure): A pymatgen Structure object
        fig (go.Figure): Figure to add the BZ to
        subplot_idx (tuple[int, int]): Row and column index of subplot

        # Surface styling
        surface_kwargs (dict): Styling for BZ surfaces.
        edge_kwargs (dict): Styling for BZ edges.
        # High symmetry point styling
        point_kwargs (dict | Literal[False]): Styling for high symmetry points.
        label_kwargs (dict): Styling for point labels.
        # High symmetry path styling
        path_kwargs (dict | Literal[False]): Styling for paths. Set to False to disable
            plotting paths.

    Returns:
        go.Figure: A plotly figure containing the first Brillouin zone
    """
    if fig is None:
        fig = go.Figure()

    # Get primitive structure first
    analyzer = SpacegroupAnalyzer(structure)
    primitive = analyzer.get_primitive_standard_structure()

    # Get reciprocal lattice vectors (scaled by 2π)
    cell = primitive.lattice.matrix
    icell = np.linalg.inv(cell).T * 2 * np.pi

    # Generate points for Voronoi construction
    px, py, pz = np.tensordot(icell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    # Create Voronoi cell
    voronoi = Voronoi(points)

    # Find central point (index 13 corresponds to [0,0,0])
    center_index = 13

    # Get BZ vertices, edges and faces
    bz_vertices = []
    bz_edges = []
    bz_faces = []

    for pid, rid in zip(voronoi.ridge_points, voronoi.ridge_vertices, strict=False):
        if (
            pid[0] == center_index or pid[1] == center_index and -1 not in rid
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

    for i, vec in enumerate(icell):
        # Add vector
        fig.add_scatter3d(
            x=[0, vec[0]],
            y=[0, vec[1]],
            z=[0, vec[2]],
            mode="lines+text",
            line=dict(color=colors[i], width=6),
            text=["", labels[i]],
            textposition="top center",
            textfont=dict(size=20, color=colors[i]),
            showlegend=False,
            row=subplot_idx[0],
            col=subplot_idx[1],
        )

    # Plot BZ faces using convex hull
    unique_vertices = np.array(vertices)
    hull = ConvexHull(unique_vertices)

    # Plot faces as a single mesh
    fig.add_mesh3d(
        x=unique_vertices[:, 0],
        y=unique_vertices[:, 1],
        z=unique_vertices[:, 2],
        i=hull.simplices[:, 0],
        j=hull.simplices[:, 1],
        k=hull.simplices[:, 2],
        showscale=False,
        row=subplot_idx[0],
        col=subplot_idx[1],
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
            row=subplot_idx[0],
            col=subplot_idx[1],
        )

    if point_kwargs is not False:
        # Add high symmetry k-points
        kpath = HighSymmKpath(structure)
        kpoints = kpath.kpath["kpoints"]
        paths = kpath.kpath["path"]

        # Convert fractional k-points to Cartesian coordinates
        cart_kpoints = {
            label: np.dot(coords, icell) for label, coords in kpoints.items()
        }

        # Plot high symmetry points
        x_coords, y_coords, z_coords, labels = [], [], [], []
        for label, coords in cart_kpoints.items():
            x_coords.append(coords[0])
            y_coords.append(coords[1])
            z_coords.append(coords[2])
            labels.append(label)

        fig.add_scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers+text",
            marker=dict(color="red", size=6) | (point_kwargs or {}),
            text=labels,
            textposition="top center",
            textfont=dict(color="black", size=12) | (label_kwargs or {}),
            name="High Symmetry Points",
            row=subplot_idx[0],
            col=subplot_idx[1],
        )

        # Add paths between high symmetry points
        if path_kwargs is not False:
            for path_points in paths:
                for i in range(len(path_points) - 1):
                    start_point = cart_kpoints[path_points[i]]
                    end_point = cart_kpoints[path_points[i + 1]]

                    fig.add_scatter3d(
                        x=[start_point[0], end_point[0]],
                        y=[start_point[1], end_point[1]],
                        z=[start_point[2], end_point[2]],
                        mode="lines",
                        line=dict(color="red", width=3, dash="dash")
                        | (path_kwargs or {}),
                        showlegend=False,
                        row=subplot_idx[0],
                        col=subplot_idx[1],
                    )

    return fig
