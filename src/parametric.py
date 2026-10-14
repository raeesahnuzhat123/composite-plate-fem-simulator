"""
Parametric Studies Engine for Laminated Composite Plates.

Executes automated trade studies:
1. Layup Comparison: [0]4, [90]4, [0/90]s, [+45/-45]s, [0/+45/-45/90]s, and custom layup.
2. Temperature Variations: deltaT = -150, -50, 0, +50, +100, +150 K.
3. Angle-Ply Sweep: theta in [0, 90] deg for [theta/-theta]s (effective CTE, Ex, failure index).
4. Load Increment to First-Ply-Failure (stress-strain curves).
5. Exports comparison CSV tables to /results directory.
6. Generates automated engineering trade-off summary.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.materials import LaminaMaterial
from src.clt import Laminate
from src.mesh import generate_rectangular_mesh
from src.thermal_fem import ThermalFEMSolver, ThermalBoundaryConditions
from src.structural_fem import StructuralFEMSolver, StructuralBoundaryConditions
from src.failure import evaluate_laminate_failure, LaminateFailureSummary


STANDARD_LAYUPS = [
    "[0]4",
    "[90]4",
    "[0/90]s",
    "[+45/-45]s",
    "[0/+45/-45/90]s"
]

DEFAULT_DELTA_T_LIST = [-150.0, -50.0, 0.0, 50.0, 100.0, 150.0]


def run_layup_comparison(material: LaminaMaterial,
                         delta_T: float = 50.0,
                         mechanical_tension: float = 100.0e6, # 100 MPa
                         user_layup: Optional[str] = None,
                         Lx: float = 0.1, Ly: float = 0.05,
                         Nx: int = 20, Ny: int = 10,
                         results_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Compare standard layups and optional user layup under combined thermal + mechanical load.
    """
    layups = list(STANDARD_LAYUPS)
    if user_layup and user_layup not in layups:
        layups.append(user_layup)

    rows = []
    detailed_results = {}

    mesh = generate_rectangular_mesh(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    for layup_str in layups:
        lam = Laminate(layup_str, material)
        eng_consts = lam.effective_engineering_constants
        alpha_eff = lam.effective_alpha_lam

        # Structural Boundary Conditions: Clamped left edge, tension on right edge
        prescribed = {}
        for n in mesh.boundary_nodes["left"]:
            prescribed[(n, 0)] = 0.0
            prescribed[(n, 1)] = 0.0

        tractions = [("right", mechanical_tension, 0.0)]
        struct_bcs = StructuralBoundaryConditions(
            prescribed_displacements=prescribed,
            tractions=tractions
        )

        # Thermal field: uniform delta_T relative to T_ref
        T_ref = 293.15
        T_nodal = np.full(mesh.num_nodes, T_ref + delta_T)

        struct_solver = StructuralFEMSolver(mesh, lam)
        struct_res = struct_solver.solve(struct_bcs, T_nodal=T_nodal, T_ref=T_ref)

        # Failure evaluation
        fail_summary = evaluate_laminate_failure(
            struct_res.ply_stresses_12,
            lam,
            mesh.element_centroids,
            criterion="Tsai-Wu"
        )

        max_disp_mm = np.max(struct_res.disp_magnitude) * 1e3
        max_sig_x_mpa = np.max(struct_res.membrane_stresses[:, 0]) / 1e6
        max_vm_mpa = np.max(struct_res.von_mises_stress) / 1e6

        row = {
            "Layup": layup_str,
            "Num_Plies": lam.num_plies,
            "Thickness_mm": lam.total_thickness * 1e3,
            "Ex_GPa": eng_consts["Ex"] / 1e9,
            "Ey_GPa": eng_consts["Ey"] / 1e9,
            "Gxy_GPa": eng_consts["Gxy"] / 1e9,
            "nu_xy": eng_consts["nu_xy"],
            "CTE_x_ppm": alpha_eff[0] * 1e6,
            "CTE_y_ppm": alpha_eff[1] * 1e6,
            "Max_Disp_mm": max_disp_mm,
            "Max_SigX_MPa": max_sig_x_mpa,
            "Max_VonMises_MPa": max_vm_mpa,
            "TsaiWu_FI": fail_summary.max_failure_index,
            "Strength_Ratio": fail_summary.min_strength_ratio,
            "FPF_Factor": fail_summary.first_ply_failure_load_factor,
            "Critical_Ply": fail_summary.critical_ply + 1,
            "Critical_Angle_deg": fail_summary.critical_angle,
            "Critical_Mode": fail_summary.critical_mode
        }
        rows.append(row)
        detailed_results[layup_str] = {
            "laminate": lam,
            "struct_results": struct_res,
            "failure_summary": fail_summary
        }

    df = pd.DataFrame(rows)

    # Export CSV if results directory provided
    if results_dir:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        csv_path = p / "parametric_layup_comparison.csv"
        df.to_csv(csv_path, index=False)

    return df, detailed_results


def run_temperature_sweep(layup_str: str,
                          material: LaminaMaterial,
                          delta_T_list: Optional[List[float]] = None,
                          mechanical_tension: float = 100.0e6,
                          Lx: float = 0.1, Ly: float = 0.05,
                          Nx: int = 20, Ny: int = 10,
                          results_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Parametric sweep of temperature change deltaT on a selected laminate.
    """
    if delta_T_list is None:
        delta_T_list = DEFAULT_DELTA_T_LIST

    mesh = generate_rectangular_mesh(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    lam = Laminate(layup_str, material)

    prescribed = {}
    for n in mesh.boundary_nodes["left"]:
        prescribed[(n, 0)] = 0.0
        prescribed[(n, 1)] = 0.0

    tractions = [("right", mechanical_tension, 0.0)]
    struct_bcs = StructuralBoundaryConditions(prescribed, tractions)
    struct_solver = StructuralFEMSolver(mesh, lam)

    T_ref = 293.15
    rows = []

    for dT in delta_T_list:
        T_nodal = np.full(mesh.num_nodes, T_ref + dT)
        struct_res = struct_solver.solve(struct_bcs, T_nodal=T_nodal, T_ref=T_ref)

        fail_summary = evaluate_laminate_failure(
            struct_res.ply_stresses_12,
            lam,
            mesh.element_centroids,
            criterion="Tsai-Wu"
        )

        rows.append({
            "DeltaT_K": dT,
            "Max_Disp_mm": np.max(struct_res.disp_magnitude) * 1e3,
            "Max_SigX_MPa": np.max(struct_res.membrane_stresses[:, 0]) / 1e6,
            "Max_VonMises_MPa": np.max(struct_res.von_mises_stress) / 1e6,
            "Failure_Index": fail_summary.max_failure_index,
            "Strength_Ratio": fail_summary.min_strength_ratio,
            "Critical_Ply": fail_summary.critical_ply + 1,
            "Critical_Angle_deg": fail_summary.critical_angle,
            "Critical_Mode": fail_summary.critical_mode
        })

    df = pd.DataFrame(rows)

    if results_dir:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        csv_path = p / "parametric_deltaT_sweep.csv"
        df.to_csv(csv_path, index=False)

    return df


def run_fiber_angle_sweep(material: LaminaMaterial,
                          angles: Optional[np.ndarray] = None,
                          delta_T: float = 50.0,
                          mechanical_tension: float = 100.0e6,
                          results_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Parametric study of fiber angle theta in angle-ply [theta/-theta]s laminate.
    Sweeps theta from 0 to 90 degrees.
    """
    if angles is None:
        angles = np.linspace(0.0, 90.0, 91)

    rows = []
    mesh = generate_rectangular_mesh(Lx=0.1, Ly=0.05, Nx=10, Ny=5)

    T_ref = 293.15
    T_nodal = np.full(mesh.num_nodes, T_ref + delta_T)

    prescribed = {}
    for n in mesh.boundary_nodes["left"]:
        prescribed[(n, 0)] = 0.0
        prescribed[(n, 1)] = 0.0
    tractions = [("right", mechanical_tension, 0.0)]
    struct_bcs = StructuralBoundaryConditions(prescribed, tractions)

    for th in angles:
        layup_angles = [th, -th, -th, th]
        lam = Laminate(layup_angles, material)
        eng = lam.effective_engineering_constants
        alpha = lam.effective_alpha_lam

        struct_solver = StructuralFEMSolver(mesh, lam)
        struct_res = struct_solver.solve(struct_bcs, T_nodal=T_nodal, T_ref=T_ref)

        fail = evaluate_laminate_failure(
            struct_res.ply_stresses_12,
            lam,
            mesh.element_centroids,
            criterion="Tsai-Wu"
        )

        rows.append({
            "Theta_deg": th,
            "Ex_GPa": eng["Ex"] / 1e9,
            "Ey_GPa": eng["Ey"] / 1e9,
            "Gxy_GPa": eng["Gxy"] / 1e9,
            "nu_xy": eng["nu_xy"],
            "CTE_x_ppm": alpha[0] * 1e6,
            "CTE_y_ppm": alpha[1] * 1e6,
            "Max_Disp_mm": np.max(struct_res.disp_magnitude) * 1e3,
            "Max_VonMises_MPa": np.max(struct_res.von_mises_stress) / 1e6,
            "TsaiWu_FI": fail.max_failure_index,
            "Strength_Ratio": fail.min_strength_ratio,
            "Critical_Mode": fail.critical_mode
        })

    df = pd.DataFrame(rows)

    if results_dir:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        csv_path = p / "parametric_angle_sweep.csv"
        df.to_csv(csv_path, index=False)

    return df


def run_load_increment_to_failure(layup_str: str,
                                  material: LaminaMaterial,
                                  base_tension: float = 10.0e6, # 10 MPa initial step
                                  delta_T: float = 0.0,
                                  max_load_factor: float = 2.0,
                                  num_steps: int = 25,
                                  results_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Increment mechanical load level up to and beyond First-Ply Failure (FPF).
    Produces stress-strain response and tracks failure index.
    """
    mesh = generate_rectangular_mesh(Lx=0.1, Ly=0.05, Nx=20, Ny=10)
    lam = Laminate(layup_str, material)

    prescribed = {}
    for n in mesh.boundary_nodes["left"]:
        prescribed[(n, 0)] = 0.0
        prescribed[(n, 1)] = 0.0

    T_ref = 293.15
    T_nodal = np.full(mesh.num_nodes, T_ref + delta_T) if delta_T != 0.0 else None

    # First determine baseline FPF load factor at base_tension
    struct_bcs = StructuralBoundaryConditions(prescribed, [("right", base_tension, 0.0)])
    struct_solver = StructuralFEMSolver(mesh, lam)
    res_base = struct_solver.solve(struct_bcs, T_nodal=T_nodal, T_ref=T_ref)
    fail_base = evaluate_laminate_failure(res_base.ply_stresses_12, lam, mesh.element_centroids, "Tsai-Wu")

    # Base strength ratio gives exact scale factor to reach FPF under pure mechanical load
    fpf_tension = base_tension * fail_base.first_ply_failure_load_factor
    target_max_tension = max(fpf_tension * max_load_factor, 100.0e6)

    applied_loads = np.linspace(0.0, target_max_tension, num_steps + 1)
    rows = []

    for load in applied_loads:
        bcs = StructuralBoundaryConditions(prescribed, [("right", load, 0.0)])
        res = struct_solver.solve(bcs, T_nodal=T_nodal, T_ref=T_ref)
        fail = evaluate_laminate_failure(res.ply_stresses_12, lam, mesh.element_centroids, "Tsai-Wu")

        # Average strain on right edge
        right_nodes = mesh.boundary_nodes["right"]
        avg_u_right = np.mean(res.displacements[right_nodes, 0])
        nominal_eps_x = avg_u_right / mesh.Lx

        rows.append({
            "Applied_Tension_MPa": load / 1e6,
            "Nominal_Strain_pct": nominal_eps_x * 100.0,
            "Max_Disp_mm": np.max(res.disp_magnitude) * 1e3,
            "Max_VonMises_MPa": np.max(res.von_mises_stress) / 1e6,
            "TsaiWu_FI": fail.max_failure_index,
            "Is_Failed": fail.max_failure_index >= 1.0,
            "Critical_Mode": fail.critical_mode
        })

    df = pd.DataFrame(rows)

    if results_dir:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        csv_path = p / "parametric_load_sweep.csv"
        df.to_csv(csv_path, index=False)

    return df


def generate_engineering_summary(layup_df: pd.DataFrame) -> str:
    """
    Generate an engineering assessment narrative identifying best and worst layups.
    """
    best_stiffness = layup_df.loc[layup_df["Ex_GPa"].idxmax()]
    best_strength = layup_df.loc[layup_df["Strength_Ratio"].idxmax()]
    worst_strength = layup_df.loc[layup_df["Strength_Ratio"].idxmin()]
    min_cte = layup_df.loc[layup_df["CTE_x_ppm"].abs().idxmin()]

    summary = f"""### Engineering Analysis & Trade-Off Summary

1. **Longitudinal Axial Stiffness (Ex)**:
   - **Optimal Layup**: `{best_stiffness['Layup']}` with $E_x = {best_stiffness['Ex_GPa']:.2f}\\text{{ GPa}}$.
   - **Mechanism**: Plies aligned along the loading axis ($0^\\circ$) utilize maximum fiber modulus $E_1$.

2. **Thermal Expansion Stability (CTE_x)**:
   - **Lowest In-Plane Distortion**: `{min_cte['Layup']}` with $\\alpha_x = {min_cte['CTE_x_ppm']:.3f}\\times 10^{{-6}}/\\text{{K}}$.
   - **Mechanism**: Fiber-dominated directions leverage the near-zero longitudinal CTE of carbon fibers ($\\alpha_1 = 0.02\\times 10^{{-6}}/\\text{{K}}$), effectively counteracting matrix expansion.

3. **Structural Margin & Failure Vulnerability**:
   - **Highest Strength Ratio (Safest)**: `{best_strength['Layup']}` ($SR = {best_strength['Strength_Ratio']:.2f}$, $FI = {best_strength['TsaiWu_FI']:.3f}$).
   - **Most Critical Layup**: `{worst_strength['Layup']}` ($SR = {worst_strength['Strength_Ratio']:.2f}$, $FI = {worst_strength['TsaiWu_FI']:.3f}$).
   - **Governing Failure Mode**: `{worst_strength['Critical_Mode']}` in Ply {worst_strength['Critical_Ply']} (Angle: {worst_strength['Critical_Angle_deg']}$^\\circ$).
   - **Failure Insight**: Transverse plies ($90^\\circ$) in cross-ply or angle-ply laminates undergo significant transverse tensile strains ($Y_t = 40\\text{{ MPa}}$), making matrix micro-cracking the primary first-ply-failure initiator well before fiber breakage.
"""
    return summary
