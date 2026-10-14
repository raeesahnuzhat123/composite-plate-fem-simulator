"""
Classical Laminate Theory (CLT) Engine.

Equations:
- Coordinate transformation angle theta (degrees to radians):
    m = cos(theta), n = sin(theta)
- Transformation matrix for stresses [T_sigma]:
    [T_sigma] = [
        [ m^2,  n^2,  2*m*n      ],
        [ n^2,  m^2, -2*m*n      ],
        [-m*n,  m*n,  m^2 - n^2  ]
    ]
- Transformation matrix for engineering strains [T_epsilon]:
    [T_epsilon] = [
        [  m^2,    n^2,     m*n       ],
        [  n^2,    m^2,    -m*n       ],
        [-2*m*n,  2*m*n,  m^2 - n^2   ]
    ]
- Note: [T_sigma]^-1(theta) = [T_sigma](-theta)
        [T_epsilon]^-1(theta) = [T_epsilon](-theta)
        [T_sigma]^T = [T_epsilon]^-1
- Transformed reduced stiffness [Qbar]:
    [Qbar] = [T_epsilon]^T * [Q] * [T_epsilon]
- Transformed thermal expansion coefficients [alpha_xy]:
    [alpha_x ]   [alpha_1 * m^2 + alpha_2 * n^2        ]
    [alpha_y ] = [alpha_1 * n^2 + alpha_2 * m^2        ]
    [alpha_xy]   [2 * (alpha_1 - alpha_2) * m * n      ]
- Laminate ABD matrices (sum over plies k=1..N):
    A_ij = sum_k Qbar_ij^(k) * (z_k - z_{k-1})
    B_ij = 0.5 * sum_k Qbar_ij^(k) * (z_k^2 - z_{k-1}^2)
    D_ij = (1/3) * sum_k Qbar_ij^(k) * (z_k^3 - z_{k-1}^3)
- Thermal resultants:
    N^T = sum_k Qbar^(k) * alpha_xy^(k) * (z_k - z_{k-1}) * deltaT
    M^T = 0.5 * sum_k Qbar^(k) * alpha_xy^(k) * (z_k^2 - z_{k-1}^2) * deltaT
- Constitutive equation:
    [N + N^T]   [A  B] [epsilon^0]
    [M + M^T] = [B  D] [kappa    ]
- Mechanical stress in ply k at coordinate z:
    epsilon_total(z) = epsilon^0 + z * kappa
    epsilon_mech(z) = epsilon_total(z) - alpha_xy^(k) * deltaT
    sigma_xy(z) = Qbar^(k) * epsilon_mech(z)
    sigma_12(z) = [T_sigma]^(k) * sigma_xy(z)
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import re
import numpy as np
from src.materials import LaminaMaterial


def parse_layup_string(layup_str: str) -> List[float]:
    """
    Parse composite layup notation into a list of ply angles in degrees.
    Supported syntaxes:
        '[0/90]s' -> [0, 90, 90, 0]
        '[0/45/-45/90]s' -> [0, 45, -45, 90, 90, -45, 45, 0]
        '[0]4' or '[0]_4' -> [0, 0, 0, 0]
        '[45/-45]2s' -> [45, -45, 45, -45, -45, 45, -45, 45]
        '0, 90, 90, 0' -> [0, 90, 90, 0]
    """
    s = layup_str.strip().replace(" ", "")
    if s.startswith("[") and "]" in s:
        idx_close = s.rfind("]")
        content = s[1:idx_close]
        suffix = s[idx_close + 1:].lower()

        # Split plies by '/' or ','
        raw_plies = [float(p) for p in re.split(r"[/,]", content) if p]

        # Parse suffix: e.g. '2s', 's', '4', '_4'
        repeat = 1
        symmetric = False

        if suffix:
            match = re.match(r"^_?(\d+)?(s)?(\d+)?$", suffix)
            if match:
                g1, g2, g3 = match.groups()
                if g1:
                    repeat = int(g1)
                if g2 == "s":
                    symmetric = True
                if g3:
                    repeat = int(g3)
            else:
                if "s" in suffix:
                    symmetric = True

        plies = raw_plies * repeat
        if symmetric:
            plies = plies + plies[::-1]
        return plies

    # Direct comma-separated or space-separated numbers
    clean_s = re.sub(r"[\[\]/]", " ", s)
    return [float(p) for p in clean_s.split() if p]


def stress_transform_matrix(theta_deg: float) -> np.ndarray:
    """
    Compute stress transformation matrix [T_sigma] for coordinate rotation by theta.
    sigma_12 = [T_sigma] * sigma_xy
    """
    rad = np.radians(theta_deg)
    m = np.cos(rad)
    n = np.sin(rad)
    return np.array([
        [m**2, n**2, 2.0 * m * n],
        [n**2, m**2, -2.0 * m * n],
        [-m * n, m * n, m**2 - n**2]
    ], dtype=np.float64)


def strain_transform_matrix(theta_deg: float) -> np.ndarray:
    """
    Compute engineering strain transformation matrix [T_epsilon] for coordinate rotation by theta.
    epsilon_12 = [T_epsilon] * epsilon_xy
    where epsilon = [epsilon_x, epsilon_y, gamma_xy]^T
    """
    rad = np.radians(theta_deg)
    m = np.cos(rad)
    n = np.sin(rad)
    return np.array([
        [m**2, n**2, m * n],
        [n**2, m**2, -m * n],
        [-2.0 * m * n, 2.0 * m * n, m**2 - n**2]
    ], dtype=np.float64)


def transformed_reduced_stiffness(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Compute transformed reduced stiffness matrix [Qbar] for ply oriented at theta_deg.
    [Qbar] = [T_epsilon]^T * [Q] * [T_epsilon]
    """
    T_eps = strain_transform_matrix(theta_deg)
    return T_eps.T @ Q @ T_eps


def transformed_thermal_expansion(alpha_mat: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Compute transformed thermal expansion coefficient vector [alpha_x, alpha_y, alpha_xy]^T.
    alpha_xy = [T_epsilon]^-1 * [alpha1, alpha2, 0]^T
    Note that [T_epsilon]^-1(theta) = [T_epsilon](-theta).
    """
    T_eps_inv = strain_transform_matrix(-theta_deg)
    return T_eps_inv @ alpha_mat


def transformed_conductivity(K_mat: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Compute transformed 2D thermal conductivity tensor in global (x, y) coordinates.
    K_xy = R * K_mat * R^T where R = [[m, -n], [n, m]]
    """
    rad = np.radians(theta_deg)
    m = np.cos(rad)
    n = np.sin(rad)
    R = np.array([
        [m, -n],
        [n,  m]
    ], dtype=np.float64)
    return R @ K_mat @ R.T


@dataclass
class Ply:
    """Individual ply in a laminate stacking sequence."""
    material: LaminaMaterial
    theta: float       # Fiber angle in degrees
    thickness: float   # Ply thickness in meters
    z_bot: float       # Bottom z-coordinate relative to laminate midplane
    z_top: float       # Top z-coordinate relative to laminate midplane

    @property
    def z_mid(self) -> float:
        return 0.5 * (self.z_top + self.z_bot)

    @property
    def Qbar(self) -> np.ndarray:
        return transformed_reduced_stiffness(self.material.Q, self.theta)

    @property
    def alpha_xy(self) -> np.ndarray:
        return transformed_thermal_expansion(self.material.alpha_mat, self.theta)

    @property
    def K_xy(self) -> np.ndarray:
        return transformed_conductivity(self.material.K_mat, self.theta)


class Laminate:
    """Classical Laminate Theory (CLT) model for a multi-ply composite laminate."""

    def __init__(self, plies_or_angles: Union[str, List[float]], default_material: LaminaMaterial,
                 thickness_per_ply: Optional[float] = None):
        """
        Initialize laminate.
        plies_or_angles: Layup string (e.g. '[0/90]s') or list of angles in degrees.
        default_material: Material assigned to plies.
        thickness_per_ply: Overriding thickness per ply in meters (optional).
        """
        if isinstance(plies_or_angles, str):
            angles = parse_layup_string(plies_or_angles)
        else:
            angles = [float(a) for a in plies_or_angles]

        self.layup_angles = angles
        t_ply = thickness_per_ply if thickness_per_ply is not None else default_material.ply_thickness
        self.num_plies = len(angles)
        self.total_thickness = self.num_plies * t_ply

        # Layup coordinates: z=0 at mid-plane
        z_start = -0.5 * self.total_thickness
        self.plies: List[Ply] = []
        current_z = z_start
        for angle in angles:
            z_bot = current_z
            z_top = current_z + t_ply
            self.plies.append(Ply(material=default_material, theta=angle, thickness=t_ply, z_bot=z_bot, z_top=z_top))
            current_z = z_top

        # Compute ABD matrices
        self.A = np.zeros((3, 3), dtype=np.float64)
        self.B = np.zeros((3, 3), dtype=np.float64)
        self.D = np.zeros((3, 3), dtype=np.float64)

        for ply in self.plies:
            Qbar = ply.Qbar
            dz = ply.z_top - ply.z_bot
            dz2 = ply.z_top**2 - ply.z_bot**2
            dz3 = ply.z_top**3 - ply.z_bot**3

            self.A += Qbar * dz
            self.B += 0.5 * Qbar * dz2
            self.D += (1.0 / 3.0) * Qbar * dz3

        self.ABD = np.block([
            [self.A, self.B],
            [self.B, self.D]
        ])

    @property
    def is_symmetric(self) -> bool:
        """Check if laminate is symmetric (B matrix near zero)."""
        return np.allclose(self.B, 0.0, atol=1e-8 * np.max(np.abs(self.A)))

    @property
    def effective_engineering_constants(self) -> dict:
        """
        Compute effective in-plane laminate engineering constants:
        Ex, Ey, Gxy, nu_xy, nu_yx.
        Using inverted in-plane compliance a = A^-1.
        """
        a = np.linalg.inv(self.A)
        H = self.total_thickness
        Ex = 1.0 / (H * a[0, 0])
        Ey = 1.0 / (H * a[1, 1])
        Gxy = 1.0 / (H * a[2, 2])
        nu_xy = -a[0, 1] / a[0, 0]
        nu_yx = -a[0, 1] / a[1, 1]
        return {
            "Ex": Ex,
            "Ey": Ey,
            "Gxy": Gxy,
            "nu_xy": nu_xy,
            "nu_yx": nu_yx
        }

    def compute_thermal_resultants(self, delta_T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thermal force resultant N_T [N/m] and moment resultant M_T [N*m/m]
        for uniform temperature change delta_T [K].
        """
        N_T = np.zeros(3, dtype=np.float64)
        M_T = np.zeros(3, dtype=np.float64)

        for ply in self.plies:
            Qbar = ply.Qbar
            alpha = ply.alpha_xy
            dz = ply.z_top - ply.z_bot
            dz2 = ply.z_top**2 - ply.z_bot**2

            N_T += (Qbar @ alpha) * dz * delta_T
            M_T += 0.5 * (Qbar @ alpha) * dz2 * delta_T

        return N_T, M_T

    @property
    def effective_alpha_lam(self) -> np.ndarray:
        """
        Effective laminate thermal expansion coefficients [alpha_x, alpha_y, alpha_xy] [1/K]
        for symmetric laminate under free thermal expansion (N=0, M=0).
        alpha_eff = A^-1 * (N_T / deltaT)
        """
        N_T_unit, _ = self.compute_thermal_resultants(delta_T=1.0)
        return np.linalg.solve(self.A, N_T_unit)

    @property
    def effective_conductivity_tensor(self) -> np.ndarray:
        """
        Effective thickness-averaged in-plane thermal conductivity tensor K_eff [W/(m*K)].
        K_eff = (1/H) * sum_k K_xy^(k) * t_k
        """
        K_sum = np.zeros((2, 2), dtype=np.float64)
        for ply in self.plies:
            K_sum += ply.K_xy * (ply.z_top - ply.z_bot)
        return K_sum / self.total_thickness

    def solve_midplane_deformation(self, N_mech: np.ndarray = None, M_mech: np.ndarray = None,
                                  delta_T: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for midplane strains epsilon0 and curvatures kappa under combined
        mechanical (N, M) and thermal (delta_T) loads.
        Constitutive relation:
            [N + N_T]   [A  B] [epsilon0]
            [M + M_T] = [B  D] [kappa   ]
        """
        if N_mech is None:
            N_mech = np.zeros(3, dtype=np.float64)
        if M_mech is None:
            M_mech = np.zeros(3, dtype=np.float64)

        N_T, M_T = self.compute_thermal_resultants(delta_T)
        F_total = np.concatenate([N_mech + N_T, M_mech + M_T])

        sol = np.linalg.solve(self.ABD, F_total)
        eps0 = sol[0:3]
        kappa = sol[3:6]
        return eps0, kappa

    def compute_ply_stresses(self, eps0: np.ndarray, kappa: np.ndarray, delta_T: float = 0.0,
                             locations: Tuple[str, ...] = ("top", "mid", "bot")) -> List[dict]:
        """
        Recover ply-by-ply strains and stresses in global (x,y) AND material (1,2) coordinates.
        Correctly subtracts free thermal strain:
            epsilon_mech = (eps0 + z*kappa) - alpha_xy * delta_T
            sigma_xy = Qbar * epsilon_mech
            sigma_12 = [T_sigma] * sigma_xy
            epsilon_12 = [T_epsilon] * epsilon_mech
        """
        results = []
        for idx, ply in enumerate(self.plies):
            ply_res = {
                "ply_index": idx,
                "theta": ply.theta,
                "material": ply.material.name,
                "z_bot": ply.z_bot,
                "z_top": ply.z_top,
                "points": {}
            }

            loc_z = {
                "bot": ply.z_bot,
                "mid": ply.z_mid,
                "top": ply.z_top
            }

            T_sigma = stress_transform_matrix(ply.theta)
            T_eps = strain_transform_matrix(ply.theta)
            alpha_xy = ply.alpha_xy

            for loc in locations:
                z = loc_z[loc]
                eps_total = eps0 + z * kappa
                eps_mech = eps_total - alpha_xy * delta_T

                sig_xy = ply.Qbar @ eps_mech
                sig_12 = T_sigma @ sig_xy
                eps_12_mech = T_eps @ eps_mech

                # Verification check: sig_12 must equal Q @ eps_12_mech
                sig_check = ply.material.Q @ eps_12_mech
                if not np.allclose(sig_12, sig_check, atol=1e-3, rtol=1e-4):
                    raise RuntimeError(f"Stress-strain inconsistency in ply {idx} at {loc}!")

                ply_res["points"][loc] = {
                    "z": z,
                    "eps_total": eps_total,
                    "eps_mech": eps_mech,
                    "sig_xy": sig_xy,
                    "sig_12": sig_12,
                    "eps_12_mech": eps_12_mech
                }

            results.append(ply_res)
        return results
