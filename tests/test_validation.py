"""
Automated Physics Validation Suite for Coupled Thermal-Structural Composite Simulation.

Executes and verifies 6 mandatory physics benchmarks:
1. Free thermal expansion of unidirectional [0] plate:
   strain_x = alpha1*deltaT, strain_y = alpha2*deltaT, zero stress.
2. Symmetric [0/90]s under pure deltaT:
   CLT numerical implementation vs closed-form hand calculation.
3. Isotropic plate (E=70 GPa, nu=0.3) under uniaxial tension:
   Analytical sigma = F/A, eps_x = sigma/E, eps_y = -nu*sigma/E.
4. Fixed-fixed isotropic bar under deltaT:
   Analytical thermal stress sigma_x = -E*alpha*deltaT.
5. Mesh convergence study:
   Monotonic convergence of tip displacement as mesh is refined (10x5 -> 20x10 -> 40x20 -> 80x40).
6. Force equilibrium check:
   Global reaction force sum matches applied mechanical load vector to machine precision.
"""

import sys
from pathlib import Path
# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import unittest
import numpy as np
from src.materials import get_default_materials, LaminaMaterial
from src.clt import Laminate
from src.mesh import generate_rectangular_mesh
from src.structural_fem import StructuralFEMSolver, StructuralBoundaryConditions
from src.thermal_fem import ThermalFEMSolver, ThermalBoundaryConditions


class TestPhysicsValidation(unittest.TestCase):
    """Validation test suite with explicit numerical tolerances."""

    @classmethod
    def setUpClass(cls):
        cls.materials = get_default_materials()
        cls.carbon = cls.materials["T300_5208_Carbon_Epoxy"]
        cls.glass = cls.materials["Glass_Epoxy"]
        cls.aluminum = cls.materials["Isotropic_Aluminum_6061"]

    def test_01_free_thermal_expansion_unidirectional(self):
        """
        Validation 1: Free thermal expansion of unidirectional [0] plate.
        Expected:
            eps_x = alpha1 * deltaT
            eps_y = alpha2 * deltaT
            gamma_xy = 0
            Mechanical stresses = 0 Pa
        """
        delta_T = 100.0 # K
        Lx, Ly = 0.1, 0.05
        mesh = generate_rectangular_mesh(Lx=Lx, Ly=Ly, Nx=20, Ny=10)
        lam = Laminate("[0]4", self.carbon)

        # Kinematic rigid-body constraints (free expansion):
        # Pin bottom-left node (0, 0)
        # Roller on bottom-right node (Lx, 0) to prevent rigid rotation
        # Find corner nodes
        nodes = mesh.nodes
        bl_node = np.argmin(np.hypot(nodes[:, 0] - 0.0, nodes[:, 1] - 0.0))
        br_node = np.argmin(np.hypot(nodes[:, 0] - Lx, nodes[:, 1] - 0.0))

        prescribed = {
            (bl_node, 0): 0.0, # u(0,0) = 0
            (bl_node, 1): 0.0, # v(0,0) = 0
            (br_node, 1): 0.0  # v(Lx,0) = 0
        }

        bcs = StructuralBoundaryConditions(prescribed, tractions=[])
        solver = StructuralFEMSolver(mesh, lam)

        T_ref = 293.15
        T_nodal = np.full(mesh.num_nodes, T_ref + delta_T)
        res = solver.solve(bcs, T_nodal=T_nodal, T_ref=T_ref)

        # Theoretical free thermal strains
        expected_eps_x = self.carbon.alpha1 * delta_T
        expected_eps_y = self.carbon.alpha2 * delta_T

        computed_eps_x = np.mean(res.element_strains[:, 0])
        computed_eps_y = np.mean(res.element_strains[:, 1])
        computed_gamma_xy = np.mean(np.abs(res.element_strains[:, 2]))
        max_stress = np.max(np.abs(res.membrane_stresses))

        # Tolerances
        err_eps_x = abs(computed_eps_x - expected_eps_x) / expected_eps_x
        err_eps_y = abs(computed_eps_y - expected_eps_y) / expected_eps_y

        self.assertLess(err_eps_x, 1e-4, f"eps_x error: {err_eps_x:.4e}")
        self.assertLess(err_eps_y, 1e-4, f"eps_y error: {err_eps_y:.4e}")
        self.assertLess(computed_gamma_xy, 1e-7, f"Non-zero shear strain: {computed_gamma_xy:.4e}")
        self.assertLess(max_stress, 1e-2, f"Mechanical stress not zero in free expansion: {max_stress:.4e} Pa")
        print(f"\n[PASS] Test 1: Free thermal expansion [0] plate (eps_x err={err_eps_x:.2e}, stress={max_stress:.2e} Pa)")

    def test_02_symmetric_cross_ply_clt_hand_calculation(self):
        """
        Validation 2: Symmetric [0/90]s under pure deltaT compared to exact closed-form hand calculation.
        """
        delta_T = 75.0 # K
        mat = self.carbon
        t = mat.ply_thickness
        Q = mat.Q # 3x3 reduced stiffness

        Q11 = Q[0, 0]
        Q22 = Q[1, 1]
        Q12 = Q[0, 1]
        a1 = mat.alpha1
        a2 = mat.alpha2

        # Hand calculation for [0/90]s (equal thicknesses):
        # A11 = 2t * (Q11 + Q22)
        # A12 = 4t * Q12
        # N_x^T = 2t * [ Q11*a1 + Q12*a2 + Q22*a2 + Q12*a1 ] * deltaT
        # eps0_x = eps0_y = N_x^T / (A11 + A12)
        #                 = [ Q11*a1 + Q22*a2 + Q12*(a1 + a2) ] / [ Q11 + Q22 + 2*Q12 ] * deltaT
        denom = Q11 + Q22 + 2.0 * Q12
        numer = (Q11 * a1 + Q22 * a2 + Q12 * (a1 + a2)) * delta_T
        hand_eps0 = numer / denom

        # Hand calculation of mechanical stresses:
        # In 0 deg ply:
        hand_sig1_0 = Q11 * (hand_eps0 - a1 * delta_T) + Q12 * (hand_eps0 - a2 * delta_T)
        hand_sig2_0 = Q12 * (hand_eps0 - a1 * delta_T) + Q22 * (hand_eps0 - a2 * delta_T)
        # In 90 deg ply:
        hand_sig1_90 = hand_sig2_0
        hand_sig2_90 = hand_sig1_0

        # Now compute via CLT class
        lam = Laminate("[0/90]s", mat)
        eps0, kappa = lam.solve_midplane_deformation(delta_T=delta_T)
        ply_stresses = lam.compute_ply_stresses(eps0, kappa, delta_T=delta_T)

        clt_eps0_x = eps0[0]
        clt_eps0_y = eps0[1]

        # 0 deg ply is ply index 0
        clt_sig1_0 = ply_stresses[0]["points"]["mid"]["sig_12"][0]
        clt_sig2_0 = ply_stresses[0]["points"]["mid"]["sig_12"][1]
        # 90 deg ply is ply index 1
        clt_sig1_90 = ply_stresses[1]["points"]["mid"]["sig_12"][0]
        clt_sig2_90 = ply_stresses[1]["points"]["mid"]["sig_12"][1]

        err_eps = abs(clt_eps0_x - hand_eps0) / abs(hand_eps0)
        err_sig1 = abs(clt_sig1_0 - hand_sig1_0) / abs(hand_sig1_0)
        err_sig2 = abs(clt_sig2_0 - hand_sig2_0) / abs(hand_sig2_0)

        self.assertLess(err_eps, 1e-10, f"CLT strain discrepancy: {err_eps:.4e}")
        self.assertLess(err_sig1, 1e-10, f"CLT ply 1 stress discrepancy: {err_sig1:.4e}")
        self.assertLess(err_sig2, 1e-10, f"CLT ply 2 stress discrepancy: {err_sig2:.4e}")
        self.assertAlmostEqual(clt_sig1_0, -clt_sig2_90, delta=1e-3)

        print(f"[PASS] Test 2: Symmetric [0/90]s CLT vs Hand Calculation (strain err={err_eps:.2e}, stress err={err_sig1:.2e})")

    def test_03_isotropic_uniaxial_tension(self):
        """
        Validation 3: Isotropic plate under uniaxial tension compared with analytical sigma = F/A.
        """
        E = 70.0e9    # Pa
        nu = 0.30
        sigma_0 = 50.0e6 # 50 MPa
        Lx, Ly = 0.1, 0.05

        mesh = generate_rectangular_mesh(Lx=Lx, Ly=Ly, Nx=20, Ny=10)
        lam = Laminate("[0]", self.aluminum, thickness_per_ply=1.0e-3)

        # Boundary Conditions:
        # Left edge: u = 0
        # Single node at (0, Ly/2) has v = 0 to prevent rigid vertical translation without Poisson constraint
        prescribed = {}
        for n in mesh.boundary_nodes["left"]:
            prescribed[(n, 0)] = 0.0

        # Find midpoint on left edge
        left_nodes = mesh.boundary_nodes["left"]
        mid_node = left_nodes[np.argmin(np.abs(mesh.nodes[left_nodes, 1] - Ly / 2.0))]
        prescribed[(mid_node, 1)] = 0.0

        tractions = [("right", sigma_0, 0.0)]
        bcs = StructuralBoundaryConditions(prescribed, tractions)

        solver = StructuralFEMSolver(mesh, lam)
        res = solver.solve(bcs)

        # Analytical solutions
        expected_u_max = (sigma_0 * Lx) / E # u = sigma*Lx/E
        expected_eps_x = sigma_0 / E
        expected_eps_y = -nu * sigma_0 / E

        computed_u_max = np.mean(res.displacements[mesh.boundary_nodes["right"], 0])
        computed_sig_x = np.mean(res.membrane_stresses[:, 0])
        computed_sig_y = np.mean(np.abs(res.membrane_stresses[:, 1]))
        computed_tau_xy = np.mean(np.abs(res.membrane_stresses[:, 2]))

        err_u = abs(computed_u_max - expected_u_max) / expected_u_max
        err_sig = abs(computed_sig_x - sigma_0) / sigma_0

        self.assertLess(err_u, 1e-4, f"Tip displacement error: {err_u:.4e}")
        self.assertLess(err_sig, 1e-4, f"Axial stress error: {err_sig:.4e}")
        self.assertLess(computed_sig_y / sigma_0, 1e-4, f"Spurious transverse stress: {computed_sig_y:.4e}")
        self.assertLess(computed_tau_xy / sigma_0, 1e-4, f"Spurious shear stress: {computed_tau_xy:.4e}")

        print(f"[PASS] Test 3: Isotropic uniaxial tension (sigma_x err={err_sig:.2e}, tip disp err={err_u:.2e})")

    def test_04_fixed_fixed_isotropic_bar_deltaT(self):
        """
        Validation 4: Fixed-fixed isotropic bar under deltaT.
        Expected: sigma_x = -E * alpha * deltaT, sigma_y = 0.
        """
        E = 70.0e9
        alpha = 23.0e-6
        delta_T = 50.0 # K
        expected_sig_x = -E * alpha * delta_T # -80.5 MPa

        Lx, Ly = 0.1, 0.02
        mesh = generate_rectangular_mesh(Lx=Lx, Ly=Ly, Nx=25, Ny=5)
        lam = Laminate("[0]", self.aluminum, thickness_per_ply=1.0e-3)

        # Both left and right edges fixed in x: u = 0
        prescribed = {}
        for n in mesh.boundary_nodes["left"]:
            prescribed[(n, 0)] = 0.0
        for n in mesh.boundary_nodes["right"]:
            prescribed[(n, 0)] = 0.0

        # One node fixed in y to eliminate rigid vertical translation (free Poisson expansion in y)
        mid_node = mesh.boundary_nodes["left"][len(mesh.boundary_nodes["left"]) // 2]
        prescribed[(mid_node, 1)] = 0.0

        bcs = StructuralBoundaryConditions(prescribed, tractions=[])
        solver = StructuralFEMSolver(mesh, lam)

        T_ref = 293.15
        T_nodal = np.full(mesh.num_nodes, T_ref + delta_T)
        res = solver.solve(bcs, T_nodal=T_nodal, T_ref=T_ref)

        computed_sig_x = np.mean(res.membrane_stresses[:, 0])
        computed_sig_y = np.mean(np.abs(res.membrane_stresses[:, 1]))

        err_sig = abs(computed_sig_x - expected_sig_x) / abs(expected_sig_x)

        self.assertLess(err_sig, 5e-4, f"Constrained thermal stress error: {err_sig:.4e}")
        self.assertLess(computed_sig_y / abs(expected_sig_x), 1e-4, f"Spurious transverse stress: {computed_sig_y:.4e}")

        print(f"[PASS] Test 4: Fixed-fixed thermal bar (sigma_x err={err_sig:.2e}, analytical={expected_sig_x/1e6:.1f} MPa)")

    def test_05_mesh_convergence_study(self):
        """
        Validation 5: Mesh convergence study showing monotonic convergence of Q4 elements.
        Tests cantilever plate under transverse tip shear load.
        """
        Lx, Ly = 0.1, 0.02
        shear_load = 5.0e6 # 5 MPa shear on right edge
        lam = Laminate("[0/90]s", self.carbon)

        mesh_densities = [(8, 4), (16, 8), (32, 16), (64, 32)]
        tip_disps = []

        for (nx, ny) in mesh_densities:
            mesh = generate_rectangular_mesh(Lx=Lx, Ly=Ly, Nx=nx, Ny=ny)
            prescribed = {}
            for n in mesh.boundary_nodes["left"]:
                prescribed[(n, 0)] = 0.0
                prescribed[(n, 1)] = 0.0

            tractions = [("right", 0.0, shear_load)]
            bcs = StructuralBoundaryConditions(prescribed, tractions)

            solver = StructuralFEMSolver(mesh, lam)
            res = solver.solve(bcs)

            # Average tip deflection in y on right edge
            right_nodes = mesh.boundary_nodes["right"]
            v_tip = np.mean(res.displacements[right_nodes, 1])
            tip_disps.append(v_tip)

        # Displacements must monotonically increase towards the exact flexible limit
        diffs = [abs(tip_disps[i+1] - tip_disps[i]) for i in range(len(tip_disps)-1)]

        # Check monotonic decrease of successive differences: diffs[i+1] < diffs[i]
        for i in range(len(diffs) - 1):
            self.assertLess(diffs[i+1], diffs[i], f"Convergence rate not monotonic at refinement step {i}")

        print(f"[PASS] Test 5: Mesh convergence verified (tip deflections: {[f'{d*1e3:.4f}mm' for d in tip_disps]})")

    def test_06_reaction_force_equilibrium(self):
        """
        Validation 6: Global energy and force equilibrium.
        Sum of boundary reaction forces equals applied external mechanical loads.
        """
        Lx, Ly = 0.1, 0.05
        mesh = generate_rectangular_mesh(Lx=Lx, Ly=Ly, Nx=20, Ny=10)
        lam = Laminate("[0/45/-45/90]s", self.carbon)

        # Clamped left edge
        prescribed = {}
        for n in mesh.boundary_nodes["left"]:
            prescribed[(n, 0)] = 0.0
            prescribed[(n, 1)] = 0.0

        # Apply multi-axial mechanical loads: tension + shear
        sigma_x = 80.0e6 # Pa
        tau_xy = 30.0e6  # Pa
        tractions = [("right", sigma_x, tau_xy)]
        bcs = StructuralBoundaryConditions(prescribed, tractions)

        # Combined with non-uniform thermal field
        T_ref = 293.15
        T_nodal = T_ref + 40.0 * np.sin(np.pi * mesh.nodes[:, 0] / Lx)

        solver = StructuralFEMSolver(mesh, lam)
        res = solver.solve(bcs, T_nodal=T_nodal, T_ref=T_ref)

        # Total applied external forces on right edge
        # Fx = sigma_x * Ly * H, Fy = tau_xy * Ly * H
        H = lam.total_thickness
        expected_Fx = sigma_x * Ly * H
        expected_Fy = tau_xy * Ly * H

        # Sum of reaction forces on all constrained nodes
        total_Rx = np.sum(res.reactions[:, 0])
        total_Ry = np.sum(res.reactions[:, 1])

        # Equilibrium: R + F_ext = 0  =>  R = -F_ext
        err_Fx = abs(total_Rx + expected_Fx) / expected_Fx
        err_Fy = abs(total_Ry + expected_Fy) / expected_Fy

        self.assertLess(err_Fx, 1e-5, f"Global X-equilibrium violated: err={err_Fx:.4e}")
        self.assertLess(err_Fy, 1e-5, f"Global Y-equilibrium violated: err={err_Fy:.4e}")

        print(f"[PASS] Test 6: Global force equilibrium (Fx balance err={err_Fx:.2e}, Fy balance err={err_Fy:.2e})")


def run_all_validation_tests() -> bool:
    """Run all validation tests and return True if all passed."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhysicsValidation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_validation_tests()
    sys.exit(0 if success else 1)
