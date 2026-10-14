"""
Material Model for Orthotropic Lamina (Plane Stress).

Equations:
- Major and minor Poisson's ratio relationship (Maxwell-Betti reciprocal theorem):
    nu21 = nu12 * (E2 / E1)
- Thermodynamic stability criterion (positive strain energy):
    nu12 * nu21 < 1.0  =>  1 - nu12 * nu21 > 0
- Reduced stiffness matrix Q (plane stress):
    Q11 = E1 / (1 - nu12 * nu21)
    Q22 = E2 / (1 - nu12 * nu21)
    Q12 = (nu12 * E2) / (1 - nu12 * nu21) = (nu21 * E1) / (1 - nu12 * nu21)
    Q66 = G12
    Q16 = Q26 = 0
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import yaml


@dataclass
class LaminaMaterial:
    """Represents an orthotropic unidirectional composite lamina or isotropic layer."""
    name: str
    E1: float            # Longitudinal Young's modulus [Pa]
    E2: float            # Transverse Young's modulus [Pa]
    G12: float           # In-plane shear modulus [Pa]
    nu12: float          # Major Poisson's ratio [-]
    alpha1: float        # Longitudinal thermal expansion coefficient [1/K]
    alpha2: float        # Transverse thermal expansion coefficient [1/K]
    k1: float            # Longitudinal thermal conductivity [W/(m*K)]
    k2: float            # Transverse thermal conductivity [W/(m*K)]
    Xt: float            # Longitudinal tensile strength [Pa]
    Xc: float            # Longitudinal compressive strength [Pa]
    Yt: float            # Transverse tensile strength [Pa]
    Yc: float            # Transverse compressive strength [Pa]
    S: float             # In-plane shear strength [Pa]
    ply_thickness: float # Nominal ply thickness [m]
    density: float = 1600.0        # Density [kg/m^3]
    specific_heat: float = 900.0   # Specific heat [J/(kg*K)]
    description: str = ""

    def __post_init__(self):
        # Enforce physical positivity
        if self.E1 <= 0 or self.E2 <= 0 or self.G12 <= 0:
            raise ValueError(f"Elastic moduli must be strictly positive: E1={self.E1}, E2={self.E2}, G12={self.G12}")
        if self.ply_thickness <= 0:
            raise ValueError(f"Ply thickness must be positive: {self.ply_thickness}")

        # Thermodynamic admissibility check
        denom = 1.0 - self.nu12 * self.nu21
        if denom <= 0:
            raise ValueError(
                f"Thermodynamic stability violated: 1 - nu12*nu21 = {denom:.4e} <= 0 "
                f"(nu12={self.nu12}, nu21={self.nu21:.4e})"
            )

    @property
    def nu21(self) -> float:
        """Minor Poisson's ratio from reciprocal relation."""
        return self.nu12 * (self.E2 / self.E1)

    @property
    def Q(self) -> np.ndarray:
        """
        Reduced stiffness matrix [Q] in material principal coordinates (1, 2).
        Units: Pa
        Order: [sigma_1, sigma_2, tau_12]^T = [Q] [epsilon_1, epsilon_2, gamma_12]^T
        """
        denom = 1.0 - self.nu12 * self.nu21
        q11 = self.E1 / denom
        q22 = self.E2 / denom
        q12 = (self.nu12 * self.E2) / denom
        q66 = self.G12

        return np.array([
            [q11, q12, 0.0],
            [q12, q22, 0.0],
            [0.0, 0.0, q66]
        ], dtype=np.float64)

    @property
    def S_compliance(self) -> np.ndarray:
        """Reduced compliance matrix [S] = [Q]^-1 in material coordinates."""
        return np.linalg.inv(self.Q)

    @property
    def alpha_mat(self) -> np.ndarray:
        """Thermal expansion coefficient vector in material principal coordinates [1/K]."""
        return np.array([self.alpha1, self.alpha2, 0.0], dtype=np.float64)

    @property
    def K_mat(self) -> np.ndarray:
        """Thermal conductivity tensor in material principal coordinates [W/(m*K)]."""
        return np.array([
            [self.k1, 0.0],
            [0.0, self.k2]
        ], dtype=np.float64)


def load_materials(config_path: Optional[str] = None) -> Dict[str, LaminaMaterial]:
    """
    Load material library from YAML configuration file.
    Falls back to default materials if path not specified or not found.
    """
    if config_path is None:
        # Default location relative to project root
        config_path = str(Path(__file__).parent.parent / "config" / "materials.yaml")

    p = Path(config_path)
    if not p.exists():
        # Fallback to hardcoded standard defaults
        return get_default_materials()

    with open(p, "r") as f:
        data = yaml.safe_load(f)

    materials = {}
    for key, props in data.items():
        mat = LaminaMaterial(
            name=props.get("name", key),
            E1=float(props["E1"]),
            E2=float(props["E2"]),
            G12=float(props["G12"]),
            nu12=float(props["nu12"]),
            alpha1=float(props["alpha1"]),
            alpha2=float(props["alpha2"]),
            k1=float(props["k1"]),
            k2=float(props["k2"]),
            Xt=float(props["Xt"]),
            Xc=float(props["Xc"]),
            Yt=float(props["Yt"]),
            Yc=float(props["Yc"]),
            S=float(props["S"]),
            ply_thickness=float(props.get("ply_thickness", 0.125e-3)),
            density=float(props.get("density", 1600.0)),
            specific_heat=float(props.get("specific_heat", 900.0)),
            description=props.get("description", "")
        )
        materials[key] = mat
    return materials


def get_default_materials() -> Dict[str, LaminaMaterial]:
    """Return default benchmark materials directly."""
    t300 = LaminaMaterial(
        name="T300/5208 Carbon/Epoxy",
        E1=181.0e9,
        E2=10.3e9,
        G12=7.17e9,
        nu12=0.28,
        alpha1=0.02e-6,
        alpha2=22.5e-6,
        k1=5.0,
        k2=0.5,
        Xt=1500.0e6,
        Xc=1500.0e6,
        Yt=40.0e6,
        Yc=246.0e6,
        S=68.0e6,
        ply_thickness=0.125e-3,
        density=1580.0,
        specific_heat=900.0,
        description="Standard high-modulus aerospace carbon/epoxy composite lamina"
    )

    glass = LaminaMaterial(
        name="E-Glass/Epoxy",
        E1=38.6e9,
        E2=8.27e9,
        G12=4.14e9,
        nu12=0.26,
        alpha1=8.6e-6,
        alpha2=22.1e-6,
        k1=0.35,
        k2=0.35,
        Xt=1062.0e6,
        Xc=610.0e6,
        Yt=31.0e6,
        Yc=118.0e6,
        S=72.0e6,
        ply_thickness=0.125e-3,
        density=1900.0,
        specific_heat=1050.0,
        description="Standard structural glass fiber reinforced polymer lamina"
    )

    aluminum = LaminaMaterial(
        name="Aluminum 6061-T6 (Isotropic Benchmark)",
        E1=70.0e9,
        E2=70.0e9,
        G12=70.0e9 / (2.0 * (1.0 + 0.3)),
        nu12=0.30,
        alpha1=23.0e-6,
        alpha2=23.0e-6,
        k1=167.0,
        k2=167.0,
        Xt=276.0e6,
        Xc=276.0e6,
        Yt=276.0e6,
        Yc=276.0e6,
        S=160.0e6,
        ply_thickness=1.0e-3,
        density=2700.0,
        specific_heat=896.0,
        description="Isotropic material benchmark for analytical verification"
    )

    return {
        "T300_5208_Carbon_Epoxy": t300,
        "Glass_Epoxy": glass,
        "Isotropic_Aluminum_6061": aluminum
    }
