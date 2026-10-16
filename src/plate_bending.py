"""
Out-of-plane plate bending: Navier double-sine series for a simply supported
rectangular laminate under uniform transverse pressure, superposed with the
free thermal warping of the laminate under a uniform Delta T.

Uses the reduced bending stiffness D* = D - B A^-1 B so unsymmetric laminates
are handled approximately; D16/D26 (bend-twist coupling) are neglected, which is
exact for cross-ply and specially orthotropic layups.
"""

import numpy as np
from src.clt import Laminate, stress_transform_matrix


def solve_navier_plate(laminate: Laminate, Lx: float, Ly: float, q0: float,
                       delta_T: float = 0.0, nx: int = 41, ny: int = 21, n_terms: int = 25):
    """
    Returns dict with grid X, Y (m), deflection W (m) and ply stresses
    sig12 of shape (num_plies, 2 [bot, top], ny, nx, 3) in Pa.
    """
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    A_inv = np.linalg.inv(laminate.A)
    Ds = laminate.D - laminate.B @ A_inv @ laminate.B

    # Pressure: Navier series (odd m, n)
    W = np.zeros_like(X)
    kx = np.zeros_like(X); ky = np.zeros_like(X); kxy = np.zeros_like(X)
    for m in range(1, n_terms + 1, 2):
        am = m * np.pi / Lx
        sx, cx = np.sin(am * X), np.cos(am * X)
        for n in range(1, n_terms + 1, 2):
            bn = n * np.pi / Ly
            denom = Ds[0, 0] * am**4 + 2 * (Ds[0, 1] + 2 * Ds[2, 2]) * am**2 * bn**2 + Ds[1, 1] * bn**4
            Wmn = 16.0 * q0 / (np.pi**2 * m * n) / denom
            sy, cy = np.sin(bn * Y), np.cos(bn * Y)
            W += Wmn * sx * sy
            # kappa = -[w_xx, w_yy, 2 w_xy]
            kx += Wmn * am**2 * sx * sy
            ky += Wmn * bn**2 * sx * sy
            kxy += -2.0 * Wmn * am * bn * cx * cy

    kappa = np.stack([kx, ky, kxy], axis=-1)              # (ny, nx, 3)
    eps0 = -kappa @ (A_inv @ laminate.B).T                # membrane strain from B coupling

    # Uniform Delta T: free expansion + warping (nonzero only for unsymmetric layups)
    eps0_T, kappa_T = laminate.solve_midplane_deformation(delta_T=delta_T)
    xc, yc = X - Lx / 2, Y - Ly / 2
    W_T = -0.5 * (kappa_T[0] * xc**2 + kappa_T[1] * yc**2 + kappa_T[2] * xc * yc)
    W_T -= W_T.min()  # rests on its lowest point
    eps0 = eps0 + eps0_T
    kappa = kappa + kappa_T

    sig12 = np.zeros((laminate.num_plies, 2, ny, nx, 3))
    for k, ply in enumerate(laminate.plies):
        T = stress_transform_matrix(ply.theta)
        for j, z in enumerate((ply.z_bot, ply.z_top)):
            eps_mech = eps0 + z * kappa - ply.alpha_xy * delta_T
            sig12[k, j] = eps_mech @ (T @ ply.Qbar).T

    return {"X": X, "Y": Y, "W": W + W_T, "sig12": sig12}


if __name__ == "__main__":
    # Check: isotropic SS square plate, w_max = 0.00406 q a^4 / D (Timoshenko)
    from src.materials import get_default_materials
    al = get_default_materials()["Isotropic_Aluminum_6061"]
    lam = Laminate([0, 0, 0, 0], al, thickness_per_ply=0.5e-3)
    a, q = 0.2, 1e3
    res = solve_navier_plate(lam, a, a, q, nx=41, ny=41)
    D = lam.D[0, 0]
    ref = 0.00406 * q * a**4 / D
    assert abs(res["W"].max() - ref) / ref < 0.01, (res["W"].max(), ref)
    print(f"OK: Navier w_max={res['W'].max()*1e3:.4f} mm vs Timoshenko {ref*1e3:.4f} mm")
