"""
2D Finite Element Method for Steady-State and Transient Heat Conduction.

Physics:
- Governing PDE:
    - div ( K_eff * grad(T) ) + Q_v = rho * c_p * dT/dt
- Laminate effective thermal conductivity tensor (tensor rotation & thickness average):
    K_eff = (1/H) * sum_k ( R(theta_k) * K_mat * R(theta_k)^T ) * t_k
- Weak form:
    integral_Omega ( grad(w)^T * K_eff * grad(T) ) * H dOmega
    + integral_Gamma_conv ( w * h_c * T ) * H dGamma
    = integral_Omega ( w * Q_v ) * H dOmega
    + integral_Gamma_conv ( w * h_c * T_inf ) * H dGamma
- Bilinear Quadrilateral (Q4) element formulation with 2x2 Gauss numerical quadrature.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from src.mesh import QuadMesh
from src.clt import Laminate


# 2x2 Gauss quadrature points and weights
GAUSS_PTS_2D = np.array([
    [-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
    [ 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
    [ 1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
    [-1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)]
], dtype=np.float64)
GAUSS_WTS_2D = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

# 1D 2-point Gauss quadrature for edges
GAUSS_PTS_1D = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
GAUSS_WTS_1D = np.array([1.0, 1.0], dtype=np.float64)


def q4_shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Q4 shape functions and derivatives at natural coordinates (xi, eta).
    Returns:
        N: (4,) shape functions [N1, N2, N3, N4]
        dN_dxi_eta: (2, 4) derivatives [[dN/dxi], [dN/deta]]
    """
    N = 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta)
    ], dtype=np.float64)

    dN_dxi_eta = 0.25 * np.array([
        [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)]
    ], dtype=np.float64)

    return N, dN_dxi_eta


@dataclass
class ThermalBoundaryConditions:
    """Thermal boundary conditions specification."""
    # Dirichlet fixed temperatures: dict mapping node index -> fixed temperature [K]
    fixed_temperatures: Dict[int, float]

    # Convection boundary condition: list of (edge_name, h_c, T_inf)
    # h_c in W/(m^2*K), T_inf in K
    convective_edges: List[Tuple[str, float, float]]

    # Volumetric heat generation function Q_v(x, y) [W/m^3]
    heat_source_fn: Optional[Callable[[float, float], float]] = None


class ThermalFEMSolver:
    """2D Q4 Finite Element Heat Conduction Solver for anisotropic composite plates."""

    def __init__(self, mesh: QuadMesh, laminate: Laminate):
        self.mesh = mesh
        self.laminate = laminate
        self.K_eff = laminate.effective_conductivity_tensor # (2, 2)
        self.thickness = laminate.total_thickness

    def compute_element_matrices(self, elem_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thermal conductivity matrix K_e (4x4) and capacitance matrix C_e (4x4) for Q4 element.
        """
        elem_node_ids = self.mesh.elements[elem_idx]
        elem_coords = self.mesh.nodes[elem_node_ids] # (4, 2)

        K_e = np.zeros((4, 4), dtype=np.float64)
        C_e = np.zeros((4, 4), dtype=np.float64)

        # Average density and specific heat from laminate
        rho_cp_avg = np.mean([p.material.density * p.material.specific_heat for p in self.laminate.plies])

        for (xi, eta), wt in zip(GAUSS_PTS_2D, GAUSS_WTS_2D):
            N, dN_dxi_eta = q4_shape_functions(xi, eta)

            # Jacobian J = dN_dxi_eta @ elem_coords -> shape (2, 2)
            J = dN_dxi_eta @ elem_coords
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(f"Negative or zero Jacobian determinant ({detJ:.4e}) in element {elem_idx}")

            invJ = np.linalg.inv(J)
            # Physical spatial derivatives B_th = invJ @ dN_dxi_eta -> shape (2, 4)
            B_th = invJ @ dN_dxi_eta

            # dOmega = detJ * wt * dxi * deta * thickness
            dV = detJ * wt * self.thickness

            K_e += B_th.T @ self.K_eff @ B_th * dV
            C_e += np.outer(N, N) * rho_cp_avg * dV

        return K_e, C_e

    def assemble_system(self, bcs: ThermalBoundaryConditions) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
        """
        Assemble global conductivity matrix K, capacitance matrix C, and thermal force vector F.
        """
        num_nodes = self.mesh.num_nodes
        num_elem = self.mesh.num_elements

        # Preallocate triplets for sparse assembly (16 entries per element)
        row_indices = []
        col_indices = []
        K_values = []
        C_values = []
        F_global = np.zeros(num_nodes, dtype=np.float64)

        for e_idx in range(num_elem):
            elem_nodes = self.mesh.elements[e_idx]
            elem_coords = self.mesh.nodes[elem_nodes]
            K_e, C_e = self.compute_element_matrices(e_idx)

            for i in range(4):
                n_i = elem_nodes[i]
                for j in range(4):
                    n_j = elem_nodes[j]
                    row_indices.append(n_i)
                    col_indices.append(n_j)
                    K_values.append(K_e[i, j])
                    C_values.append(C_e[i, j])

            # Integrate volumetric heat source if provided
            if bcs.heat_source_fn is not None:
                for (xi, eta), wt in zip(GAUSS_PTS_2D, GAUSS_WTS_2D):
                    N, dN_dxi_eta = q4_shape_functions(xi, eta)
                    J = dN_dxi_eta @ elem_coords
                    detJ = np.linalg.det(J)
                    x_phys, y_phys = N @ elem_coords
                    q_val = bcs.heat_source_fn(x_phys, y_phys)
                    dV = detJ * wt * self.thickness
                    F_global[elem_nodes] += N * q_val * dV

        # Convective boundary edges
        for (edge_name, h_c, T_inf) in bcs.convective_edges:
            edges = self.mesh.boundary_edges.get(edge_name, [])
            for (n1, n2) in edges:
                p1 = self.mesh.nodes[n1]
                p2 = self.mesh.nodes[n2]
                L_edge = np.linalg.norm(p2 - p1)

                # 1D Gauss quadrature along edge
                for xi, wt in zip(GAUSS_PTS_1D, GAUSS_WTS_1D):
                    # Linear 1D shape functions: N_1 = 0.5*(1-xi), N_2 = 0.5*(1+xi)
                    N_1d = np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])
                    dGamma = (L_edge / 2.0) * wt * self.thickness

                    K_edge = np.outer(N_1d, N_1d) * h_c * dGamma
                    F_edge = N_1d * (h_c * T_inf) * dGamma

                    for i, ni in enumerate([n1, n2]):
                        F_global[ni] += F_edge[i]
                        for j, nj in enumerate([n1, n2]):
                            row_indices.append(ni)
                            col_indices.append(nj)
                            K_values.append(K_edge[i, j])
                            C_values.append(0.0)

        K_global = sp.coo_matrix((K_values, (row_indices, col_indices)), shape=(num_nodes, num_nodes)).tocsr()
        C_global = sp.coo_matrix((C_values, (row_indices, col_indices)), shape=(num_nodes, num_nodes)).tocsr()

        return K_global, C_global, F_global

    def solve_steady_state(self, bcs: ThermalBoundaryConditions) -> np.ndarray:
        """
        Solve steady-state temperature field T(x, y) [K].
        K_global * T = F_global with Dirichlet boundary conditions.
        """
        K_global, _, F_global = self.assemble_system(bcs)
        num_nodes = self.mesh.num_nodes

        # Apply Dirichlet boundary conditions
        fixed_nodes = list(bcs.fixed_temperatures.keys())
        fixed_values = np.array(list(bcs.fixed_temperatures.values()), dtype=np.float64)

        if len(fixed_nodes) == 0:
            raise ValueError("No fixed temperature or Dirichlet BC provided: system is singular under pure Neumann")

        # Symmetric Dirichlet elimination
        # F_mod = F - K[:, fixed] * T_fixed
        T_sol = np.zeros(num_nodes, dtype=np.float64)
        T_sol[fixed_nodes] = fixed_values

        F_mod = F_global - K_global @ T_sol

        # Zero out rows and columns of fixed nodes, put 1.0 on diagonal
        free_mask = np.ones(num_nodes, dtype=bool)
        free_mask[fixed_nodes] = False
        free_indices = np.where(free_mask)[0]

        K_free = K_global[free_indices, :][:, free_indices]
        F_free = F_mod[free_indices]

        T_free = spla.spsolve(K_free, F_free)
        T_sol[free_indices] = T_free

        return T_sol

    def solve_transient(self, bcs: ThermalBoundaryConditions, T_initial: np.ndarray,
                        dt: float, total_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve transient heat conduction using Implicit Euler time integration:
        (C + dt * K) * T^{n+1} = C * T^n + dt * F^{n+1}
        Returns:
            times: 1D array of time stamps
            T_history: (num_steps, num_nodes) temperature histories
        """
        K_global, C_global, F_global = self.assemble_system(bcs)
        num_nodes = self.mesh.num_nodes

        fixed_nodes = list(bcs.fixed_temperatures.keys())
        fixed_values = np.array(list(bcs.fixed_temperatures.values()), dtype=np.float64)

        free_mask = np.ones(num_nodes, dtype=bool)
        free_mask[fixed_nodes] = False
        free_indices = np.where(free_mask)[0]

        # Effective system matrix: A_sys = C + dt * K
        A_sys = C_global + dt * K_global
        A_free = A_sys[free_indices, :][:, free_indices]
        solver = spla.factorized(A_free.tocsc())

        num_steps = int(np.ceil(total_time / dt))
        times = np.linspace(0.0, total_time, num_steps + 1)
        T_history = np.zeros((num_steps + 1, num_nodes), dtype=np.float64)
        T_history[0] = T_initial.copy()

        current_T = T_initial.copy()
        for step in range(num_steps):
            # RHS = C * current_T + dt * F_global
            RHS = C_global @ current_T + dt * F_global

            # Prescribe fixed values
            T_next = np.zeros(num_nodes, dtype=np.float64)
            T_next[fixed_nodes] = fixed_values

            RHS_mod = RHS - A_sys @ T_next
            T_next[free_indices] = solver(RHS_mod[free_indices])

            current_T = T_next
            T_history[step + 1] = current_T

        return times, T_history

    def compute_heat_flux(self, T_nodal: np.ndarray) -> np.ndarray:
        """
        Compute in-plane heat flux vector q = -K_eff * grad(T) [W/m^2] at element centroids.
        Returns array of shape (num_elements, 2).
        """
        num_elem = self.mesh.num_elements
        fluxes = np.zeros((num_elem, 2), dtype=np.float64)

        for e_idx in range(num_elem):
            elem_nodes = self.mesh.elements[e_idx]
            elem_coords = self.mesh.nodes[elem_nodes]
            T_elem = T_nodal[elem_nodes]

            # At centroid (xi=0, eta=0)
            _, dN_dxi_eta = q4_shape_functions(0.0, 0.0)
            J = dN_dxi_eta @ elem_coords
            invJ = np.linalg.inv(J)
            B_th = invJ @ dN_dxi_eta

            grad_T = B_th @ T_elem # [dT/dx, dT/dy]^T
            fluxes[e_idx] = -self.K_eff @ grad_T

        return fluxes
