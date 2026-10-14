"""
2D Finite Element Method for Plane Stress Structural Mechanics of Composite Plates.

Physics:
- Governing Equations (Membrane equilibrium under thermal + mechanical loads):
    div(N) + b = 0
    N = A * epsilon^0 - N^T
- Classical Laminate Theory (CLT) membrane stiffness matrix:
    A = sum_k Qbar^(k) * t_k  [N/m]
- Thermal force resultant:
    N^T(x, y) = sum_k ( Qbar^(k) * alpha_xy^(k) * t_k ) * deltaT(x, y)
- Element stiffness matrix (Q4, 2x2 Gauss integration):
    K_e = integral_Omega B^T * A * B dOmega  [N/m]
- Consistent thermal load vector:
    f_e^th = integral_Omega B^T * N^T(x, y) dOmega  [N]
- Mechanical boundary tractions (tension, shear, pressure):
    f_edge = integral_Gamma N_1D^T * traction * H dGamma  [N]
- Ply-by-ply stress recovery in global (x,y) and principal material axes (1,2):
    epsilon_mech^(k) = epsilon^0 - alpha_xy^(k) * deltaT
    sigma_xy^(k) = Qbar^(k) * epsilon_mech^(k)
    sigma_12^(k) = [T_sigma]^(k) * sigma_xy^(k)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from src.mesh import QuadMesh
from src.clt import Laminate, stress_transform_matrix, strain_transform_matrix
from src.thermal_fem import q4_shape_functions, GAUSS_PTS_2D, GAUSS_WTS_2D, GAUSS_PTS_1D, GAUSS_WTS_1D


@dataclass
class StructuralBoundaryConditions:
    """Structural boundary conditions and applied mechanical loads."""
    # Prescribed nodal displacements: dict mapping (node_idx, dof_dir) -> prescribed_val
    # dof_dir: 0 for u (x-disp), 1 for v (y-disp)
    prescribed_displacements: Dict[Tuple[int, int], float]

    # Boundary tractions: list of (edge_name, tx, ty) where tx, ty are in Pa (stress)
    tractions: List[Tuple[str, float, float]]


@dataclass
class StructuralResults:
    """Complete structural simulation outputs."""
    displacements: np.ndarray      # (num_nodes, 2) array of [u, v] in meters
    disp_magnitude: np.ndarray     # (num_nodes,) displacement magnitude |u| in meters
    element_strains: np.ndarray    # (num_elem, 3) midplane strains [eps_x, eps_y, gamma_xy]
    membrane_forces: np.ndarray    # (num_elem, 3) force resultants [N_x, N_y, N_xy] in N/m
    membrane_stresses: np.ndarray  # (num_elem, 3) thickness-averaged stresses [sig_x, sig_y, tau_xy] in Pa
    von_mises_stress: np.ndarray   # (num_elem,) von Mises stress in Pa
    ply_stresses_12: np.ndarray    # (num_elem, num_plies, 3) material stresses [sig_1, sig_2, tau_12] in Pa
    ply_stresses_xy: np.ndarray    # (num_elem, num_plies, 3) global stresses [sig_x, sig_y, tau_xy] in Pa
    reactions: np.ndarray          # (num_nodes, 2) reaction forces [R_x, R_y] in N
    nodal_von_mises: np.ndarray    # (num_nodes,) extrapolated von Mises stress for plotting
    nodal_sig_x: np.ndarray        # (num_nodes,) extrapolated sigma_x
    nodal_sig_y: np.ndarray        # (num_nodes,) extrapolated sigma_y
    nodal_tau_xy: np.ndarray       # (num_nodes,) extrapolated tau_xy


class StructuralFEMSolver:
    """2D Q4 Finite Element Structural Solver for coupled thermo-mechanical composite plates."""

    def __init__(self, mesh: QuadMesh, laminate: Laminate):
        self.mesh = mesh
        self.laminate = laminate
        self.A = laminate.A                      # (3, 3) membrane stiffness [N/m]
        self.thickness = laminate.total_thickness # Total thickness [m]

    def compute_b_matrix(self, dN_dxi_eta: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
        Compute 3x8 strain-displacement matrix B for Q4 element.
        Order of DOFs: [u1, v1, u2, v2, u3, v3, u4, v4]
        Order of strains: [epsilon_x, epsilon_y, gamma_xy]^T
        """
        invJ = np.linalg.inv(J)
        dN_dxy = invJ @ dN_dxi_eta # (2, 4) -> [[dN1/dx, dN2/dx, ...], [dN1/dy, dN2/dy, ...]]

        B = np.zeros((3, 8), dtype=np.float64)
        for a in range(4):
            dN_dx = dN_dxy[0, a]
            dN_dy = dN_dxy[1, a]

            B[0, 2 * a]     = dN_dx
            B[1, 2 * a + 1] = dN_dy
            B[2, 2 * a]     = dN_dy
            B[2, 2 * a + 1] = dN_dx

        return B

    def compute_element_stiffness(self, elem_idx: int) -> np.ndarray:
        """
        Compute 8x8 element stiffness matrix K_e = integral B^T * A * B dOmega.
        """
        elem_node_ids = self.mesh.elements[elem_idx]
        elem_coords = self.mesh.nodes[elem_node_ids]

        K_e = np.zeros((8, 8), dtype=np.float64)

        for (xi, eta), wt in zip(GAUSS_PTS_2D, GAUSS_WTS_2D):
            _, dN_dxi_eta = q4_shape_functions(xi, eta)
            J = dN_dxi_eta @ elem_coords
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(f"Non-positive Jacobian determinant in element {elem_idx}: {detJ}")

            B = self.compute_b_matrix(dN_dxi_eta, J)
            # Area differential dA = detJ * wt * dxi * deta
            dA = detJ * wt
            K_e += B.T @ self.A @ B * dA

        return K_e

    def compute_element_thermal_load(self, elem_idx: int, T_nodal: np.ndarray,
                                     T_ref: float = 293.15) -> np.ndarray:
        """
        Compute consistent element thermal load vector:
        f_e^th = integral B^T * N^T(xi, eta) dOmega
        where N^T(xi, eta) = sum_k (Qbar^(k) * alpha_xy^(k) * t_k) * (T(xi, eta) - T_ref)
        """
        elem_node_ids = self.mesh.elements[elem_idx]
        elem_coords = self.mesh.nodes[elem_node_ids]
        T_elem = T_nodal[elem_node_ids]

        f_th_e = np.zeros(8, dtype=np.float64)

        # Thermal stiffness resultant coefficient: S_th = sum_k Qbar^(k) * alpha_xy^(k) * t_k [N/(m*K)]
        S_th = np.zeros(3, dtype=np.float64)
        for ply in self.laminate.plies:
            S_th += (ply.Qbar @ ply.alpha_xy) * (ply.z_top - ply.z_bot)

        for (xi, eta), wt in zip(GAUSS_PTS_2D, GAUSS_WTS_2D):
            N, dN_dxi_eta = q4_shape_functions(xi, eta)
            J = dN_dxi_eta @ elem_coords
            detJ = np.linalg.det(J)

            B = self.compute_b_matrix(dN_dxi_eta, J)
            T_gauss = np.dot(N, T_elem)
            delta_T = T_gauss - T_ref

            N_T = S_th * delta_T # Thermal resultant at Gauss point [N/m]
            dA = detJ * wt
            f_th_e += B.T @ N_T * dA

        return f_th_e

    def assemble_system(self, bcs: StructuralBoundaryConditions,
                        T_nodal: Optional[np.ndarray] = None,
                        T_ref: float = 293.15) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Assemble global stiffness matrix K (2*N x 2*N) and total load vector F (2*N).
        """
        num_nodes = self.mesh.num_nodes
        num_elem = self.mesh.num_elements
        num_dof = 2 * num_nodes

        # 8x8 = 64 entries per element
        rows = []
        cols = []
        K_vals = []
        F_global = np.zeros(num_dof, dtype=np.float64)

        for e_idx in range(num_elem):
            elem_nodes = self.mesh.elements[e_idx]
            K_e = self.compute_element_stiffness(e_idx)

            # Element DOF indices: [2*n1, 2*n1+1, 2*n2, 2*n2+1, ...]
            elem_dofs = np.zeros(8, dtype=np.int64)
            for a in range(4):
                elem_dofs[2 * a]     = 2 * elem_nodes[a]
                elem_dofs[2 * a + 1] = 2 * elem_nodes[a] + 1

            for i in range(8):
                di = elem_dofs[i]
                for j in range(8):
                    dj = elem_dofs[j]
                    rows.append(di)
                    cols.append(dj)
                    K_vals.append(K_e[i, j])

            # Thermal load
            if T_nodal is not None:
                f_th_e = self.compute_element_thermal_load(e_idx, T_nodal, T_ref)
                F_global[elem_dofs] += f_th_e

        # Mechanical surface tractions
        for (edge_name, tx_stress, ty_stress) in bcs.tractions:
            edges = self.mesh.boundary_edges.get(edge_name, [])
            # Traction per unit length: t_force = stress * thickness [N/m]
            tx_force = tx_stress * self.thickness
            ty_force = ty_stress * self.thickness

            for (n1, n2) in edges:
                p1 = self.mesh.nodes[n1]
                p2 = self.mesh.nodes[n2]
                L_edge = np.linalg.norm(p2 - p1)

                # Consistent nodal loads for constant traction on linear edge = 0.5 * t * L_edge
                Fx_node = 0.5 * tx_force * L_edge
                Fy_node = 0.5 * ty_force * L_edge

                F_global[2 * n1]     += Fx_node
                F_global[2 * n1 + 1] += Fy_node
                F_global[2 * n2]     += Fx_node
                F_global[2 * n2 + 1] += Fy_node

        K_global = sp.coo_matrix((K_vals, (rows, cols)), shape=(num_dof, num_dof)).tocsr()
        return K_global, F_global

    def solve(self, bcs: StructuralBoundaryConditions,
              T_nodal: Optional[np.ndarray] = None,
              T_ref: float = 293.15) -> StructuralResults:
        """
        Solve coupled structural system K * u = F and post-process full stress/strain fields.
        """
        num_nodes = self.mesh.num_nodes
        num_dof = 2 * num_nodes
        num_elem = self.mesh.num_elements

        K_global, F_global = self.assemble_system(bcs, T_nodal, T_ref)

        # Convert prescribed boundary conditions to DOF index mapping
        prescribed_dof_indices = []
        prescribed_values = []
        for (n_idx, dof_dir), val in bcs.prescribed_displacements.items():
            dof = 2 * n_idx + dof_dir
            prescribed_dof_indices.append(dof)
            prescribed_values.append(val)

        prescribed_dof_indices = np.array(prescribed_dof_indices, dtype=np.int64)
        prescribed_values = np.array(prescribed_values, dtype=np.float64)

        if len(prescribed_dof_indices) == 0:
            raise ValueError("No kinematic boundary conditions specified: plate has rigid body modes!")

        u_total = np.zeros(num_dof, dtype=np.float64)
        u_total[prescribed_dof_indices] = prescribed_values

        free_mask = np.ones(num_dof, dtype=bool)
        free_mask[prescribed_dof_indices] = False
        free_indices = np.where(free_mask)[0]

        # F_free = F_global - K @ u_prescribed
        RHS_free = F_global[free_indices] - (K_global[free_indices, :][:, prescribed_dof_indices] @ prescribed_values)
        K_free = K_global[free_indices, :][:, free_indices]

        u_free = spla.spsolve(K_free, RHS_free)
        u_total[free_indices] = u_free

        # Reactions R = K * u - F_ext
        R_total = K_global @ u_total - F_global
        reactions = np.column_stack([R_total[0::2], R_total[1::2]])

        # Displacement field
        u_disp = u_total[0::2]
        v_disp = u_total[1::2]
        displacements = np.column_stack([u_disp, v_disp])
        disp_mag = np.hypot(u_disp, v_disp)

        # Stress and strain recovery at element centroids
        elem_strains = np.zeros((num_elem, 3), dtype=np.float64)
        membrane_forces = np.zeros((num_elem, 3), dtype=np.float64)
        membrane_stresses = np.zeros((num_elem, 3), dtype=np.float64)
        von_mises = np.zeros(num_elem, dtype=np.float64)

        num_plies = self.laminate.num_plies
        ply_stresses_12 = np.zeros((num_elem, num_plies, 3), dtype=np.float64)
        ply_stresses_xy = np.zeros((num_elem, num_plies, 3), dtype=np.float64)

        for e_idx in range(num_elem):
            elem_nodes = self.mesh.elements[e_idx]
            elem_coords = self.mesh.nodes[elem_nodes]

            elem_dofs = np.zeros(8, dtype=np.int64)
            for a in range(4):
                elem_dofs[2 * a]     = 2 * elem_nodes[a]
                elem_dofs[2 * a + 1] = 2 * elem_nodes[a] + 1
            u_e = u_total[elem_dofs]

            # Evaluate at element centroid (xi=0, eta=0)
            N_c, dN_dxi_eta = q4_shape_functions(0.0, 0.0)
            J = dN_dxi_eta @ elem_coords
            B = self.compute_b_matrix(dN_dxi_eta, J)

            eps0 = B @ u_e # Total midplane strain [eps_x, eps_y, gamma_xy]
            elem_strains[e_idx] = eps0

            # Local temperature at centroid
            if T_nodal is not None:
                T_c = np.dot(N_c, T_nodal[elem_nodes])
                delta_T = T_c - T_ref
            else:
                delta_T = 0.0

            # Thermal resultant N_T
            N_T_c, _ = self.laminate.compute_thermal_resultants(delta_T)
            N_c_total = self.A @ eps0 - N_T_c
            membrane_forces[e_idx] = N_c_total

            # Thickness-averaged membrane stress
            sig_avg = N_c_total / self.thickness
            membrane_stresses[e_idx] = sig_avg

            # Von Mises: sqrt(sig_x^2 - sig_x*sig_y + sig_y^2 + 3*tau_xy^2)
            vm = np.sqrt(max(0.0, sig_avg[0]**2 - sig_avg[0]*sig_avg[1] + sig_avg[1]**2 + 3.0*sig_avg[2]**2))
            von_mises[e_idx] = vm

            # Ply-by-ply stresses
            for k_idx, ply in enumerate(self.laminate.plies):
                T_sigma = stress_transform_matrix(ply.theta)
                # Mechanical strain = eps0 - alpha_xy * delta_T
                eps_mech_k = eps0 - ply.alpha_xy * delta_T
                sig_xy_k = ply.Qbar @ eps_mech_k
                sig_12_k = T_sigma @ sig_xy_k

                ply_stresses_xy[e_idx, k_idx] = sig_xy_k
                ply_stresses_12[e_idx, k_idx] = sig_12_k

        # Recover stresses at element corner nodes and average onto nodes
        nodal_sig_x = np.zeros(num_nodes, dtype=np.float64)
        nodal_sig_y = np.zeros(num_nodes, dtype=np.float64)
        nodal_tau_xy = np.zeros(num_nodes, dtype=np.float64)
        nodal_vm = np.zeros(num_nodes, dtype=np.float64)
        node_counts = np.zeros(num_nodes, dtype=np.float64)

        corner_natural = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

        for e_idx in range(num_elem):
            elem_nodes = self.mesh.elements[e_idx]
            elem_coords = self.mesh.nodes[elem_nodes]

            elem_dofs = np.zeros(8, dtype=np.int64)
            for a in range(4):
                elem_dofs[2 * a]     = 2 * elem_nodes[a]
                elem_dofs[2 * a + 1] = 2 * elem_nodes[a] + 1
            u_e = u_total[elem_dofs]

            for a, (xi_a, eta_a) in enumerate(corner_natural):
                n_id = elem_nodes[a]
                N_a, dN_a = q4_shape_functions(xi_a, eta_a)
                J_a = dN_a @ elem_coords
                B_a = self.compute_b_matrix(dN_a, J_a)
                eps_corner = B_a @ u_e

                if T_nodal is not None:
                    T_corner = np.dot(N_a, T_nodal[elem_nodes])
                    dT_corner = T_corner - T_ref
                else:
                    dT_corner = 0.0

                N_T_corner, _ = self.laminate.compute_thermal_resultants(dT_corner)
                sig_corner = (self.A @ eps_corner - N_T_corner) / self.thickness
                vm_corner = np.sqrt(max(0.0, sig_corner[0]**2 - sig_corner[0]*sig_corner[1] + sig_corner[1]**2 + 3.0*sig_corner[2]**2))

                nodal_sig_x[n_id] += sig_corner[0]
                nodal_sig_y[n_id] += sig_corner[1]
                nodal_tau_xy[n_id] += sig_corner[2]
                nodal_vm[n_id] += vm_corner
                node_counts[n_id] += 1.0

        nodal_sig_x /= np.maximum(node_counts, 1.0)
        nodal_sig_y /= np.maximum(node_counts, 1.0)
        nodal_tau_xy /= np.maximum(node_counts, 1.0)
        nodal_vm /= np.maximum(node_counts, 1.0)

        return StructuralResults(
            displacements=displacements,
            disp_magnitude=disp_mag,
            element_strains=elem_strains,
            membrane_forces=membrane_forces,
            membrane_stresses=membrane_stresses,
            von_mises_stress=von_mises,
            ply_stresses_12=ply_stresses_12,
            ply_stresses_xy=ply_stresses_xy,
            reactions=reactions,
            nodal_von_mises=nodal_vm,
            nodal_sig_x=nodal_sig_x,
            nodal_sig_y=nodal_sig_y,
            nodal_tau_xy=nodal_tau_xy
        )
