"""
2D Finite Element Mesh Generation for Bilinear Quadrilateral (Q4) Elements.

Features:
1. Structured rectangular plate mesh generator (Nx x Ny elements).
2. Structured 4-block pure-quadrilateral mesh with a central circular hole.
3. Node, element, and boundary classification (nodes, edge segments, and normals).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class QuadMesh:
    """Represents a 2D mesh of 4-node bilinear quadrilaterals (Q4)."""
    nodes: np.ndarray             # (num_nodes, 2) array of (x, y) coordinates
    elements: np.ndarray          # (num_elements, 4) array of node indices (CCW order)
    boundary_nodes: Dict[str, np.ndarray]  # Map: 'left', 'right', 'bottom', 'top', 'hole' -> node indices
    boundary_edges: Dict[str, List[Tuple[int, int]]] # Edge segment pairs (node1, node2) per boundary
    Lx: float                     # Length in x [m]
    Ly: float                     # Width in y [m]
    has_hole: bool = False
    hole_radius: float = 0.0
    hole_center: Tuple[float, float] = (0.0, 0.0)

    @property
    def num_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def num_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def element_centroids(self) -> np.ndarray:
        """Compute centroid (xc, yc) for each Q4 element."""
        # elements: (Ne, 4), nodes[elements]: (Ne, 4, 2)
        return np.mean(self.nodes[self.elements], axis=1)

    @property
    def element_areas(self) -> np.ndarray:
        """Compute area of each Q4 element via shoelace formula."""
        elem_coords = self.nodes[self.elements] # (Ne, 4, 2)
        x = elem_coords[:, :, 0]
        y = elem_coords[:, :, 1]
        area = 0.5 * np.abs(
            (x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0]) +
            (x[:, 1]*y[:, 2] - x[:, 2]*y[:, 1]) +
            (x[:, 2]*y[:, 3] - x[:, 3]*y[:, 2]) +
            (x[:, 3]*y[:, 0] - x[:, 0]*y[:, 3])
        )
        return area


def generate_rectangular_mesh(Lx: float = 0.1, Ly: float = 0.05,
                              Nx: int = 40, Ny: int = 20) -> QuadMesh:
    """
    Generate a structured grid of Q4 elements on a rectangular plate [0, Lx] x [0, Ly].

    Node ordering is column-by-row:
        node_id = i + j * (Nx + 1)
        where i in 0..Nx, j in 0..Ny
    Element connectivity is counter-clockwise:
        n1 = (i, j), n2 = (i+1, j), n3 = (i+1, j+1), n4 = (i, j+1)
    """
    x = np.linspace(0.0, Lx, Nx + 1)
    y = np.linspace(0.0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y) # X, Y shape: (Ny+1, Nx+1)

    nodes = np.column_stack([X.ravel(), Y.ravel()])
    num_nodes = (Nx + 1) * (Ny + 1)

    elements = []
    for j in range(Ny):
        for i in range(Nx):
            n1 = i + j * (Nx + 1)
            n2 = (i + 1) + j * (Nx + 1)
            n3 = (i + 1) + (j + 1) * (Nx + 1)
            n4 = i + (j + 1) * (Nx + 1)
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements, dtype=np.int64)

    # Boundary node identification
    tol = 1e-7 * min(Lx, Ly)
    left_nodes = np.where(np.abs(nodes[:, 0] - 0.0) < tol)[0]
    right_nodes = np.where(np.abs(nodes[:, 0] - Lx) < tol)[0]
    bottom_nodes = np.where(np.abs(nodes[:, 1] - 0.0) < tol)[0]
    top_nodes = np.where(np.abs(nodes[:, 1] - Ly) < tol)[0]

    # Boundary edges (for line integrals of surface traction or convection)
    # Order edges consistently
    left_edges = []
    for j in range(Ny):
        n_bot = 0 + j * (Nx + 1)
        n_top = 0 + (j + 1) * (Nx + 1)
        left_edges.append((n_bot, n_top))

    right_edges = []
    for j in range(Ny):
        n_bot = Nx + j * (Nx + 1)
        n_top = Nx + (j + 1) * (Nx + 1)
        right_edges.append((n_bot, n_top))

    bottom_edges = []
    for i in range(Nx):
        n_l = i
        n_r = i + 1
        bottom_edges.append((n_l, n_r))

    top_edges = []
    for i in range(Nx):
        n_l = i + Ny * (Nx + 1)
        n_r = (i + 1) + Ny * (Nx + 1)
        top_edges.append((n_l, n_r))

    boundary_nodes = {
        "left": left_nodes,
        "right": right_nodes,
        "bottom": bottom_nodes,
        "top": top_nodes
    }

    boundary_edges = {
        "left": left_edges,
        "right": right_edges,
        "bottom": bottom_edges,
        "top": top_edges
    }

    return QuadMesh(
        nodes=nodes,
        elements=elements,
        boundary_nodes=boundary_nodes,
        boundary_edges=boundary_edges,
        Lx=Lx,
        Ly=Ly,
        has_hole=False
    )


def generate_plate_with_hole_mesh(Lx: float = 0.1, Ly: float = 0.05,
                                  radius: float = 0.005,
                                  N_circum: int = 16,
                                  N_radial: int = 12) -> QuadMesh:
    """
    Generate an all-quadrilateral conformal mesh for a rectangular plate with a central circular hole.
    Uses a 4-sector mapping (Right, Top, Left, Bottom) transitioning smoothly from the inner circle
    to the outer rectangular boundaries.

    Args:
        Lx: Plate length in x [m]
        Ly: Plate width in y [m]
        radius: Central circular hole radius [m] (must be < min(Lx, Ly)/2)
        N_circum: Number of quad elements along each of the 4 sectors (total 4*N_circum circumferential)
        N_radial: Number of quad elements in the radial direction
    """
    cx, cy = 0.5 * Lx, 0.5 * Ly
    r_max_allowed = 0.45 * min(Lx, Ly)
    if radius >= r_max_allowed:
        radius = r_max_allowed

    # 4 sectors: Right (-45 to +45 deg), Top (45 to 135 deg), Left (135 to 225 deg), Bottom (225 to 315 deg)
    # Total circumferential nodes = 4 * N_circum
    total_circum = 4 * N_circum

    # We will build a structured grid in (circumferential index theta_idx, radial index r_idx)
    # theta_idx in 0..total_circum, r_idx in 0..N_radial
    # r_idx = 0 is the hole surface (r = radius), r_idx = N_radial is the outer rectangle boundary.

    # Radial spacing with gentle geometric bias towards the hole boundary for resolving stress concentrations
    # bias factor eta in [0, 1]
    s = np.linspace(0.0, 1.0, N_radial + 1)
    # Quadratic grading: denser near hole
    radial_frac = s**1.3

    node_coords = []
    # Build circumferential angles spanning 0 to 2*pi
    # Let theta_0 = -pi/4 so sector 0 is Right, sector 1 is Top, sector 2 is Left, sector 3 is Bottom
    angles = []
    for sector in range(4):
        theta_start = -np.pi / 4.0 + sector * (np.pi / 2.0)
        theta_end = theta_start + np.pi / 2.0
        # N_circum subdivisions in this sector
        th = np.linspace(theta_start, theta_end, N_circum, endpoint=False)
        angles.extend(th)
    angles = np.array(angles)

    # For each angle, find the outer boundary distance from center (cx, cy)
    # The outer boundary is a rectangle [-Lx/2, Lx/2] x [-Ly/2, Ly/2] relative to center
    hx, hy = 0.5 * Lx, 0.5 * Ly

    outer_pts = np.zeros((total_circum, 2))
    inner_pts = np.zeros((total_circum, 2))

    for idx, th in enumerate(angles):
        cos_t = np.cos(th)
        sin_t = np.sin(th)

        # Inner circular point
        inner_pts[idx] = [cx + radius * cos_t, cy + radius * sin_t]

        # Ray from center: (cx + r*cos_t, cy + r*sin_t)
        # Find intersection with rectangle edges:
        # x = +-hx => r_x = +-hx / cos_t
        # y = +-hy => r_y = +-hy / sin_t
        cand_r = []
        if np.abs(cos_t) > 1e-10:
            rx1 = hx / cos_t
            if rx1 > 0:
                cand_r.append(rx1)
            rx2 = -hx / cos_t
            if rx2 > 0:
                cand_r.append(rx2)
        if np.abs(sin_t) > 1e-10:
            ry1 = hy / sin_t
            if ry1 > 0:
                cand_r.append(ry1)
            ry2 = -hy / sin_t
            if ry2 > 0:
                cand_r.append(ry2)

        r_outer = min(cand_r)
        outer_pts[idx] = [cx + r_outer * cos_t, cy + r_outer * sin_t]

    # Generate nodes: 2D grid of size (total_circum, N_radial + 1)
    # node_grid[th_idx, r_idx] = global_node_index
    grid_nodes = np.zeros((total_circum, N_radial + 1, 2), dtype=np.float64)
    for c_idx in range(total_circum):
        p_in = inner_pts[c_idx]
        p_out = outer_pts[c_idx]
        for r_idx in range(N_radial + 1):
            f = radial_frac[r_idx]
            grid_nodes[c_idx, r_idx] = (1.0 - f) * p_in + f * p_out

    # Flatten nodes
    nodes = []
    node_id_map = {}
    node_counter = 0

    for c_idx in range(total_circum):
        for r_idx in range(N_radial + 1):
            nodes.append(grid_nodes[c_idx, r_idx])
            node_id_map[(c_idx, r_idx)] = node_counter
            node_counter += 1

    nodes = np.array(nodes, dtype=np.float64)

    # Build Q4 elements (ordered counter-clockwise: r_dir then theta_dir)
    elements = []
    for c_idx in range(total_circum):
        c_next = (c_idx + 1) % total_circum
        for r_idx in range(N_radial):
            n1 = node_id_map[(c_idx, r_idx)]
            n2 = node_id_map[(c_idx, r_idx + 1)]
            n3 = node_id_map[(c_next, r_idx + 1)]
            n4 = node_id_map[(c_next, r_idx)]
            elements.append([n1, n2, n3, n4])

    elements = np.array(elements, dtype=np.int64)

    # Identify boundaries
    tol = 1e-6 * min(Lx, Ly)
    left_nodes = np.where(np.abs(nodes[:, 0] - 0.0) < tol)[0]
    right_nodes = np.where(np.abs(nodes[:, 0] - Lx) < tol)[0]
    bottom_nodes = np.where(np.abs(nodes[:, 1] - 0.0) < tol)[0]
    top_nodes = np.where(np.abs(nodes[:, 1] - Ly) < tol)[0]

    # Hole nodes: r_idx == 0
    hole_nodes = np.array([node_id_map[(c_idx, 0)] for c_idx in range(total_circum)], dtype=np.int64)

    # Edge segments
    # Outer boundaries are at r_idx = N_radial
    outer_edges = []
    for c_idx in range(total_circum):
        c_next = (c_idx + 1) % total_circum
        outer_edges.append((node_id_map[(c_idx, N_radial)], node_id_map[(c_next, N_radial)]))

    left_edges = []
    right_edges = []
    bottom_edges = []
    top_edges = []
    hole_edges = []

    for c_idx in range(total_circum):
        c_next = (c_idx + 1) % total_circum
        n_a = node_id_map[(c_idx, 0)]
        n_b = node_id_map[(c_next, 0)]
        hole_edges.append((n_a, n_b))

    for e in outer_edges:
        n1, n2 = e
        p1, p2 = nodes[n1], nodes[n2]
        mid = 0.5 * (p1 + p2)
        if np.abs(mid[0] - 0.0) < tol:
            left_edges.append(e)
        elif np.abs(mid[0] - Lx) < tol:
            right_edges.append(e)
        elif np.abs(mid[1] - 0.0) < tol:
            bottom_edges.append(e)
        elif np.abs(mid[1] - Ly) < tol:
            top_edges.append(e)

    boundary_nodes = {
        "left": left_nodes,
        "right": right_nodes,
        "bottom": bottom_nodes,
        "top": top_nodes,
        "hole": hole_nodes
    }

    boundary_edges = {
        "left": left_edges,
        "right": right_edges,
        "bottom": bottom_edges,
        "top": top_edges,
        "hole": hole_edges
    }

    return QuadMesh(
        nodes=nodes,
        elements=elements,
        boundary_nodes=boundary_nodes,
        boundary_edges=boundary_edges,
        Lx=Lx,
        Ly=Ly,
        has_hole=True,
        hole_radius=radius,
        hole_center=(cx, cy)
    )
