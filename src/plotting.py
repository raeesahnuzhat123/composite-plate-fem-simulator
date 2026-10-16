"""
Visualization and Plotting Utilities for Coupled Thermal-Structural Composite Simulation.

Features:
- Deformed mesh contour maps (with adjustable deformation scale factor) for:
    - Temperature field T(x, y) [K or degC]
    - Displacement magnitude |u| [mm]
    - In-plane membrane stresses sigma_x, sigma_y, tau_xy, von Mises [MPa]
    - Failure Index map with highlighted failure zones (FI >= 1.0)
- Through-thickness ply-by-ply stress step-plots (sigma_1, sigma_2, tau_12) with ply boundary markers
- Parametric comparison charts (Layup trade-offs, DeltaT sweeps, angle-ply curves, stress-strain curves)
- Strict unit annotations on all axes and colorbars
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from src.mesh import QuadMesh
from src.clt import Laminate
from src.structural_fem import StructuralResults
from src.failure import LaminateFailureSummary

# Ensure matplotlib cache directory is writable
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_fcm"


def q4_to_triangles(elements: np.ndarray) -> np.ndarray:
    """
    Split Q4 quadrilateral elements into pairs of triangles for matplotlib Triangulation.
    Q4 [n1, n2, n3, n4] -> Tri1 [n1, n2, n3], Tri2 [n1, n3, n4]
    """
    num_elem = elements.shape[0]
    triangles = np.zeros((num_elem * 2, 3), dtype=np.int64)

    for i in range(num_elem):
        n1, n2, n3, n4 = elements[i]
        triangles[2 * i]     = [n1, n2, n3]
        triangles[2 * i + 1] = [n1, n3, n4]

    return triangles


def plot_field_on_deformed_mesh(mesh: QuadMesh,
                                nodal_field: np.ndarray,
                                displacements: Optional[np.ndarray] = None,
                                scale_factor: float = 1.0,
                                title: str = "Contour Plot",
                                cbar_label: str = "Value",
                                cmap: str = "viridis",
                                vmin: Optional[float] = None,
                                vmax: Optional[float] = None,
                                show_mesh: bool = True,
                                highlight_failure: bool = False,
                                failure_threshold: float = 1.0,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Render a color-coded continuous contour plot on the (optionally deformed) Q4 mesh.

    Args:
        mesh: QuadMesh geometry
        nodal_field: (num_nodes,) field values at nodes
        displacements: (num_nodes, 2) displacements [u, v] in meters
        scale_factor: Magnification scale factor for deformed coordinates
        title: Figure title
        cbar_label: Colorbar label with physical units
        cmap: Colormap name
        save_path: Optional path to save high-res figure (PNG/PDF)
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Compute coordinates (in millimeters for intuitive display)
    if displacements is not None and scale_factor > 0:
        coords_m = mesh.nodes + scale_factor * displacements
    else:
        coords_m = mesh.nodes

    coords_mm = coords_m * 1e3
    x_mm = coords_mm[:, 0]
    y_mm = coords_mm[:, 1]

    triangles = q4_to_triangles(mesh.elements)
    triang = mtri.Triangulation(x_mm, y_mm, triangles)

    level_min = float(vmin if vmin is not None else np.min(nodal_field))
    level_max = float(vmax if vmax is not None else np.max(nodal_field))
    if level_min > level_max:
        raise ValueError("vmin must be less than or equal to vmax")
    if level_min == level_max:
        padding = max(abs(level_min) * 0.02, 1e-6)  # constant field: readable colorbar, no offset notation
        level_min -= padding
        level_max += padding
    levels = np.linspace(level_min, level_max, 25)

    tcf = ax.tricontourf(triang, nodal_field, levels=levels, cmap=cmap, extend="both")
    cbar = fig.colorbar(tcf, ax=ax, orientation="vertical", pad=0.03, aspect=25)
    cbar.set_label(cbar_label, fontsize=10, fontweight="bold")

    # Overlay wireframe mesh if requested
    if show_mesh:
        ax.triplot(triang, color="k", linewidth=0.25, alpha=0.35)

    # Highlight failure zone (FI >= failure_threshold) if requested
    if highlight_failure and np.any(nodal_field >= failure_threshold):
        cs = ax.tricontour(triang, nodal_field, levels=[failure_threshold], colors="red", linewidths=1.8)
        ax.clabel(cs, fmt=f"Failure Zone (FI={failure_threshold})", fontsize=8)

    scale_text = f" (Deform Scale: {scale_factor:.1f}x)" if (displacements is not None and scale_factor > 0) else ""
    ax.set_title(f"{title}{scale_text}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("x [mm]", fontsize=10)
    ax.set_ylabel("y [mm]", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_through_thickness_stresses(laminate: Laminate,
                                    ply_stresses_12: np.ndarray,
                                    elem_idx: int = 0,
                                    title: Optional[str] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ply-by-ply stress distribution (sigma_1, sigma_2, tau_12) through the laminate thickness.

    Args:
        laminate: Laminate object
        ply_stresses_12: (num_elements, num_plies, 3) material stresses in Pa
        elem_idx: Element index to inspect (typically critical element)
        save_path: Optional path to save image
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    z_vals = []
    sig1_vals = []
    sig2_vals = []
    tau12_vals = []

    # Construct step profile across each ply
    for k_idx, ply in enumerate(laminate.plies):
        z_bot_mm = ply.z_bot * 1e3
        z_top_mm = ply.z_top * 1e3

        s1 = ply_stresses_12[elem_idx, k_idx, 0] / 1e6   # MPa
        s2 = ply_stresses_12[elem_idx, k_idx, 1] / 1e6   # MPa
        t12 = ply_stresses_12[elem_idx, k_idx, 2] / 1e6  # MPa

        # Bottom and top of ply
        z_vals.extend([z_bot_mm, z_top_mm])
        sig1_vals.extend([s1, s1])
        sig2_vals.extend([s2, s2])
        tau12_vals.extend([t12, t12])

        # Draw ply interface line
        ax.axhline(z_top_mm, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
        # Ply label on right
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != 1.0 else 0.0,
                0.5 * (z_bot_mm + z_top_mm),
                f" Ply {k_idx+1}: [{ply.theta:.0f}°]",
                fontsize=8, va="center", color="#333333")

    ax.plot(sig1_vals, z_vals, label=r"$\sigma_1$ (Fiber Direction)", color="#1f77b4", linewidth=2.0)
    ax.plot(sig2_vals, z_vals, label=r"$\sigma_2$ (Transverse Direction)", color="#ff7f0e", linewidth=2.0)
    ax.plot(tau12_vals, z_vals, label=r"$\tau_{12}$ (In-Plane Shear)", color="#2ca02c", linewidth=2.0)

    ax.axvline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, label="Midplane (z=0)")

    ax.set_ylabel("Through-Thickness Coordinate z [mm]", fontsize=10, fontweight="bold")
    ax.set_xlabel("Material Stress [MPa]", fontsize=10, fontweight="bold")
    if title is None:
        title = f"Through-Thickness Stress Distribution (Element {elem_idx})"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_parametric_layup_comparison(layup_df: pd.DataFrame,
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Bar chart comparing axial modulus Ex, Strength Ratio, and Failure Index across layups.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    layups = layup_df["Layup"]
    x = np.arange(len(layups))
    width = 0.35

    # Plot 1: Engineering Stiffness Ex and Ey
    ax1.bar(x - width/2, layup_df["Ex_GPa"], width, label=r"Axial $E_x$", color="#2b5c8f")
    ax1.bar(x + width/2, layup_df["Ey_GPa"], width, label=r"Transverse $E_y$", color="#d95f02")
    ax1.set_ylabel("Effective Modulus [GPa]", fontsize=10, fontweight="bold")
    ax1.set_title("In-Plane Stiffness Comparison", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layups, rotation=15, ha="right")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Plot 2: Strength Ratio and Failure Index
    ax2.bar(x - width/2, layup_df["Strength_Ratio"], width, label="Strength Ratio (SR)", color="#2ca02c")
    ax2.bar(x + width/2, layup_df["TsaiWu_FI"], width, label="Tsai-Wu FI", color="#d62728")
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Failure Threshold (1.0)")
    ax2.set_ylabel("Safety Metric [-]", fontsize=10, fontweight="bold")
    ax2.set_title("Failure Margins (Tsai-Wu)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layups, rotation=15, ha="right")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_parametric_deltaT(temp_df: pd.DataFrame,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot max deflection and max von Mises stress vs temperature change deltaT.
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=150)

    color = "#1f77b4"
    ax1.set_xlabel(r"Temperature Change $\Delta T$ [K]", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Max Deflection [mm]", color=color, fontsize=10, fontweight="bold")
    line1 = ax1.plot(temp_df["DeltaT_K"], temp_df["Max_Disp_mm"], color=color, marker="o", linewidth=2.0, label="Max Deflection")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color = "#d62728"
    ax2.set_ylabel(r"Max Von Mises Stress [MPa]", color=color, fontsize=10, fontweight="bold")
    line2 = ax2.plot(temp_df["DeltaT_K"], temp_df["Max_VonMises_MPa"], color=color, marker="s", linewidth=2.0, linestyle="--", label="Max Von Mises")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=9)
    ax1.set_title(r"Thermo-Mechanical Response vs Temperature Variation $\Delta T$", fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_parametric_angle_sweep(angle_df: pd.DataFrame,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot effective axial modulus Ex and longitudinal CTE alpha_x vs fiber angle theta.
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=150)

    color = "#2b5c8f"
    ax1.set_xlabel(r"Fiber Orientation Angle $\theta$ [deg] in $[\theta/-\theta]_s$", fontsize=10, fontweight="bold")
    ax1.set_ylabel(r"Effective Axial Modulus $E_x$ [GPa]", color=color, fontsize=10, fontweight="bold")
    l1 = ax1.plot(angle_df["Theta_deg"], angle_df["Ex_GPa"], color=color, linewidth=2.2, label=r"$E_x$ Modulus")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color = "#e6550d"
    ax2.set_ylabel(r"Thermal Expansion $\alpha_x$ [$10^{-6}$/K]", color=color, fontsize=10, fontweight="bold")
    l2 = ax2.plot(angle_df["Theta_deg"], angle_df["CTE_x_ppm"], color=color, linewidth=2.2, linestyle="--", label=r"CTE $\alpha_x$")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=9)
    ax1.set_title(r"Stiffness and Thermal Expansion vs Fiber Orientation $[\theta/-\theta]_s$", fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_stress_strain_curve(load_df: pd.DataFrame,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot applied stress vs nominal strain curve highlighting First-Ply Failure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(load_df["Nominal_Strain_pct"], load_df["Applied_Tension_MPa"],
            color="#1f77b4", linewidth=2.2, label="Nominal Response")

    # Locate first ply failure
    failed_idx = load_df["Is_Failed"].idxmax() if np.any(load_df["Is_Failed"]) else None
    if failed_idx is not None and load_df.loc[failed_idx, "Is_Failed"]:
        fpf_row = load_df.loc[failed_idx]
        ax.plot(fpf_row["Nominal_Strain_pct"], fpf_row["Applied_Tension_MPa"],
                marker="*", markersize=14, color="red", label=f"First-Ply Failure (FPF) - {fpf_row['Critical_Mode']}")
        ax.axvline(fpf_row["Nominal_Strain_pct"], color="red", linestyle=":", alpha=0.7)
        ax.axhline(fpf_row["Applied_Tension_MPa"], color="red", linestyle=":", alpha=0.7)

    ax.set_xlabel(r"Nominal Tensile Strain $\epsilon_x$ [%]", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Applied Tensile Stress $\sigma_x$ [MPa]", fontsize=10, fontweight="bold")
    ax.set_title("Stress-Strain Curve up to First-Ply Failure", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig
