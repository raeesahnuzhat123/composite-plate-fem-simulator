"""
Failure Theories and Criteria Engine for Orthotropic Composite Laminates.

Implemented Criteria:
1. Maximum Stress Criterion (independent failure modes: fiber tension/compression, matrix tension/compression, shear)
2. Tsai-Hill Criterion (interactive energy-based quadratic formulation)
3. Tsai-Wu Criterion (tensor polynomial with Tsai-Hahn F12 interaction parameter)

Calculates:
- Failure Index (FI) for every ply and element
- Strength Ratio (SR)
- Critical ply, critical element, and governing failure mode
- First-Ply-Failure (FPF) load factor lambda_FPF
- Failure zone map (FI >= 1.0)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from src.materials import LaminaMaterial
from src.clt import Laminate


@dataclass
class FailureEvaluation:
    """Detailed failure assessment for an element or ply."""
    failure_index: float        # FI (failure if >= 1.0)
    strength_ratio: float       # SR (failure if <= 1.0)
    failure_mode: str           # Governing mode description
    criterion: str              # Evaluated criterion name
    is_failed: bool             # True if FI >= 1.0


@dataclass
class LaminateFailureSummary:
    """Laminate-level failure evaluation across all elements and plies."""
    criterion: str
    max_failure_index: float
    min_strength_ratio: float
    first_ply_failure_load_factor: float
    critical_element: int
    critical_ply: int
    critical_angle: float
    critical_mode: str
    critical_coords: Tuple[float, float]
    element_failure_indices: np.ndarray # (num_elements,) maximum FI over plies
    element_failure_mask: np.ndarray    # (num_elements,) True if FI >= 1.0
    ply_failure_indices: np.ndarray     # (num_elements, num_plies)


def eval_max_stress(sig12: np.ndarray, mat: LaminaMaterial) -> FailureEvaluation:
    """
    Evaluate Maximum Stress Failure Criterion for a given stress state [sig_1, sig_2, tau_12].
    """
    s1, s2, t12 = sig12[0], sig12[1], sig12[2]

    f_1t = max(0.0, s1) / mat.Xt
    f_1c = max(0.0, -s1) / mat.Xc
    f_2t = max(0.0, s2) / mat.Yt
    f_2c = max(0.0, -s2) / mat.Yc
    f_12 = abs(t12) / mat.S

    modes = [
        (f_1t, "Fiber Tension"),
        (f_1c, "Fiber Compression"),
        (f_2t, "Matrix Tension"),
        (f_2c, "Matrix Compression"),
        (f_12, "In-Plane Shear")
    ]

    max_fi, governing_mode = max(modes, key=lambda x: x[0])
    sr = 1.0 / max_fi if max_fi > 1e-12 else 1e9

    return FailureEvaluation(
        failure_index=max_fi,
        strength_ratio=sr,
        failure_mode=governing_mode,
        criterion="Maximum Stress",
        is_failed=(max_fi >= 1.0)
    )


def eval_tsai_hill(sig12: np.ndarray, mat: LaminaMaterial) -> FailureEvaluation:
    """
    Evaluate Tsai-Hill Criterion:
    (sig1 / X)^2 - (sig1 * sig2) / X^2 + (sig2 / Y)^2 + (tau12 / S)^2 = FI_TH
    """
    s1, s2, t12 = sig12[0], sig12[1], sig12[2]

    X = mat.Xt if s1 >= 0 else mat.Xc
    Y = mat.Yt if s2 >= 0 else mat.Yc
    S = mat.S

    fi = (s1 / X)**2 - (s1 * s2) / (X**2) + (s2 / Y)**2 + (t12 / S)**2
    sr = 1.0 / np.sqrt(fi) if fi > 1e-12 else 1e9

    # Identify dominant component
    t_fiber = (s1 / X)**2
    t_matrix = (s2 / Y)**2
    t_shear = (t12 / S)**2

    if t_fiber >= max(t_matrix, t_shear):
        mode = "Fiber Dominant (" + ("Tension" if s1 >= 0 else "Compression") + ")"
    elif t_matrix >= max(t_fiber, t_shear):
        mode = "Matrix Dominant (" + ("Tension" if s2 >= 0 else "Compression") + ")"
    else:
        mode = "Shear Dominant"

    return FailureEvaluation(
        failure_index=fi,
        strength_ratio=sr,
        failure_mode=mode,
        criterion="Tsai-Hill",
        is_failed=(fi >= 1.0)
    )


def eval_tsai_wu(sig12: np.ndarray, mat: LaminaMaterial) -> FailureEvaluation:
    """
    Evaluate Tsai-Wu Criterion with Tsai-Hahn interaction parameter F12:
    F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + F66*t12^2 + 2*F12*s1*s2 = FI
    """
    s1, s2, t12 = sig12[0], sig12[1], sig12[2]

    F1 = 1.0 / mat.Xt - 1.0 / mat.Xc
    F2 = 1.0 / mat.Yt - 1.0 / mat.Yc
    F11 = 1.0 / (mat.Xt * mat.Xc)
    F22 = 1.0 / (mat.Yt * mat.Yc)
    F66 = 1.0 / (mat.S**2)
    # Tsai-Hahn assumption for stable ellipse
    F12 = -0.5 * np.sqrt(F11 * F22)

    fi_direct = (F1 * s1 + F2 * s2 +
                 F11 * s1**2 + F22 * s2**2 + F66 * t12**2 +
                 2.0 * F12 * s1 * s2)

    # Solve for strength ratio R: a*R^2 + b*R - 1 = 0
    a = F11 * s1**2 + F22 * s2**2 + F66 * t12**2 + 2.0 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2

    if a > 1e-14:
        disc = b**2 + 4.0 * a
        if disc >= 0:
            sr = (-b + np.sqrt(disc)) / (2.0 * a)
        else:
            sr = 1e9
    elif abs(b) > 1e-14:
        sr = 1.0 / b if b > 0 else 1e9
    else:
        sr = 1e9

    # Governing mode identification
    ratio_fiber = abs(s1) / (mat.Xt if s1 >= 0 else mat.Xc)
    ratio_matrix = abs(s2) / (mat.Yt if s2 >= 0 else mat.Yc)
    ratio_shear = abs(t12) / mat.S

    if ratio_fiber >= max(ratio_matrix, ratio_shear):
        mode = "Fiber (" + ("Tension" if s1 >= 0 else "Compression") + ")"
    elif ratio_matrix >= max(ratio_fiber, ratio_shear):
        mode = "Matrix (" + ("Tension" if s2 >= 0 else "Compression") + ")"
    else:
        mode = "In-Plane Shear"

    # FI is 1 / SR
    fi = 1.0 / sr if sr > 1e-12 else 1e9

    return FailureEvaluation(
        failure_index=fi,
        strength_ratio=sr,
        failure_mode=mode,
        criterion="Tsai-Wu",
        is_failed=(fi >= 1.0)
    )


def evaluate_laminate_failure(ply_stresses_12: np.ndarray,
                              laminate: Laminate,
                              element_centroids: np.ndarray,
                              criterion: str = "Tsai-Wu") -> LaminateFailureSummary:
    """
    Evaluate failure across all elements and plies for the selected criterion.

    Args:
        ply_stresses_12: (num_elements, num_plies, 3) array of material stresses [sig1, sig2, tau12]
        laminate: Laminate object with ply materials and angles
        element_centroids: (num_elements, 2) array of element centroids
        criterion: 'Max Stress', 'Tsai-Hill', or 'Tsai-Wu'
    """
    num_elem, num_plies, _ = ply_stresses_12.shape
    ply_fi_matrix = np.zeros((num_elem, num_plies), dtype=np.float64)
    ply_sr_matrix = np.zeros((num_elem, num_plies), dtype=np.float64)
    ply_mode_matrix = [["" for _ in range(num_plies)] for _ in range(num_elem)]

    crit_norm = criterion.lower().replace("-", " ").replace("_", " ")

    for e_idx in range(num_elem):
        for k_idx, ply in enumerate(laminate.plies):
            sig12 = ply_stresses_12[e_idx, k_idx]
            mat = ply.material

            if "max" in crit_norm or "stress" in crit_norm:
                res = eval_max_stress(sig12, mat)
            elif "hill" in crit_norm:
                res = eval_tsai_hill(sig12, mat)
            else: # Default Tsai-Wu
                res = eval_tsai_wu(sig12, mat)

            ply_fi_matrix[e_idx, k_idx] = res.failure_index
            ply_sr_matrix[e_idx, k_idx] = res.strength_ratio
            ply_mode_matrix[e_idx][k_idx] = res.failure_mode

    # Max failure index per element
    elem_max_fi = np.max(ply_fi_matrix, axis=1)
    elem_fail_mask = elem_max_fi >= 1.0

    # Overall critical point
    crit_elem, crit_ply = np.unravel_index(np.argmax(ply_fi_matrix), ply_fi_matrix.shape)
    max_fi = float(ply_fi_matrix[crit_elem, crit_ply])
    min_sr = float(ply_sr_matrix[crit_elem, crit_ply])
    fpf_factor = min_sr

    crit_angle = float(laminate.plies[crit_ply].theta)
    crit_mode = ply_mode_matrix[crit_elem][crit_ply]
    crit_coords = (float(element_centroids[crit_elem, 0]), float(element_centroids[crit_elem, 1]))

    return LaminateFailureSummary(
        criterion=criterion,
        max_failure_index=max_fi,
        min_strength_ratio=min_sr,
        first_ply_failure_load_factor=fpf_factor,
        critical_element=int(crit_elem),
        critical_ply=int(crit_ply),
        critical_angle=crit_angle,
        critical_mode=crit_mode,
        critical_coords=crit_coords,
        element_failure_indices=elem_max_fi,
        element_failure_mask=elem_fail_mask,
        ply_failure_indices=ply_fi_matrix
    )
