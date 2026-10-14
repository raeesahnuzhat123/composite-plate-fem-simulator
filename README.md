# Coupled Thermal-Structural Composite Plate FEM Simulation

A first-principles, high-fidelity Python simulation of the coupled thermal + structural response of fiber-reinforced composite laminates under thermo-mechanical loading.

Includes an custom 2D Finite Element Method (FEM) solver for heat conduction and plane stress mechanics, a full Classical Laminate Theory (CLT) engine, multiple failure criteria, automated parametric trade studies, and an interactive Streamlit application.

---

## Key Features

1. **Orthotropic Material Modeling**:
   - Plane stress lamina constitutive relations ($E_1, E_2, G_{12}, \nu_{12}, \nu_{21}$).
   - Principal thermal expansion ($\alpha_1, \alpha_2$) and conductivity ($k_1, k_2$) tensors.
   - Built-in material library: **T300/5208 Carbon/Epoxy**, **E-Glass/Epoxy**, and **Isotropic Aluminum 6061** benchmark.
2. **Classical Laminate Theory (CLT)**:
   - Reduced stiffness matrix $\mathbf{Q}$ and transformed stiffness $\bar{\mathbf{Q}}(\theta)$.
   - Full Laminate $[\mathbf{A}, \mathbf{B}, \mathbf{D}]$ matrices.
   - Transformed thermal expansion vector $\boldsymbol{\alpha}_{xy}$ and thermal force/moment resultants $\mathbf{N}^T, \mathbf{M}^T$.
   - Effective in-plane laminate engineering constants ($E_x, E_y, G_{xy}, \nu_{xy}, \bar{\boldsymbol{\alpha}}_{lam}$).
   - Ply-by-ply mechanical stress recovery with exact thermal strain subtraction: $\boldsymbol{\sigma} = \bar{\mathbf{Q}}(\boldsymbol{\epsilon} - \boldsymbol{\alpha}\Delta T)$.
3. **Thermal 2D FEM Solver**:
   - Anisotropic 2D steady-state and transient heat conduction ($-\nabla \cdot (\mathbf{K}_{eff} \nabla T) = Q_v$).
   - Bilinear quadrilateral (Q4) isoparametric elements with $2 \times 2$ Gauss quadrature.
   - Selectable boundary conditions: fixed edge temperatures (Dirichlet), localized Gaussian hot spots, edge convection (Robin).
4. **Structural 2D FEM Solver**:
   - Plane stress membrane formulation using laminate membrane stiffness $\mathbf{A}$.
   - Coupled consistent nodal thermal load vector: $\mathbf{f}_e^{th} = \int \mathbf{B}^T \mathbf{N}^T(x,y)\,d\Omega$.
   - Selectable mechanical loads: uniaxial tension, biaxial tension, in-plane shear, edge pressure.
   - Structural boundary conditions: clamped edge, simply supported / roller, kinematically constrained free expansion.
   - Conformal all-quad structured mesh around central circular holes for stress concentration validation ($K_t \approx 3.0$).
5. **Failure Analysis Engine**:
   - Three failure theories: **Maximum Stress**, **Tsai-Hill**, and **Tsai-Wu** (Tsai-Hahn interaction $F_{12}$).
   - Failure index (FI) and strength ratio (SR) for every ply in every element.
   - Automatic detection of critical ply, critical location, and governing failure mode (Fiber Tension/Compression, Matrix Cracking, Shear).
   - First-Ply-Failure (FPF) load factor $\lambda_{FPF}$.
6. **Automated Parametric Studies**:
   - Layup trade studies: `[0]4`, `[90]4`, `[0/90]s`, `[+45/-45]s`, `[0/+45/-45/90]s`, and custom user layups.
   - Temperature variation sweep: $\Delta T \in [-150, +150]\text{ K}$.
   - Angle-ply fiber orientation sweep: $[\theta/-\theta]_s$ for $\theta \in [0^\circ, 90^\circ]$.
   - Load increment curves tracing non-linear nominal stress-strain response to First-Ply Failure.
   - Automatic export of comparison CSV tables and figures to `/results`.
7. **Automated Physics Validation Suite**:
   - 6 automated benchmarks comparing code output against exact analytical solutions, reporting `PASS`/`FAIL`.

---

## Project Structure

```
├── config/
│   └── materials.yaml          # Material properties database in SI units
├── src/
│   ├── __init__.py
│   ├── materials.py            # Lamina properties & constitutive tensor [Q]
│   ├── clt.py                  # Classical Laminate Theory engine ([A, B, D], Qbar, stresses)
│   ├── mesh.py                 # Structured Q4 mesh generator (rectangular & with circular hole)
│   ├── thermal_fem.py          # 2D anisotropic heat conduction solver (Q4 elements)
│   ├── structural_fem.py       # 2D structural plane stress solver (coupled thermal-mechanical)
│   ├── failure.py              # Max Stress, Tsai-Hill, and Tsai-Wu failure criteria
│   ├── parametric.py           # Parametric sweep engine and CSV exporter
│   └── plotting.py             # Matplotlib contour maps on deformed mesh & step-plots
├── tests/
│   ├── __init__.py
│   └── test_validation.py      # Automated physics verification suite (6 benchmarks)
├── results/                    # Auto-generated CSVs and plot artifacts
├── app.py                      # Interactive Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # Technical documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Recommended: virtual environment

### 1. Clone or Navigate to the Project Directory
```bash
git clone https://github.com/<your-username>/<repository>.git
cd <repository>
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Run Automated Physics Validation Suite
Execute the 6 analytical verification tests:
```bash
python3 tests/test_validation.py
```
Expected output:
```
[PASS] Test 1: Free thermal expansion [0] plate (eps_x err=1.42e-14, stress=7.86e-07 Pa)
[PASS] Test 2: Symmetric [0/90]s CLT vs Hand Calculation (strain err=1.19e-16, stress err=2.34e-16)
[PASS] Test 3: Isotropic uniaxial tension (sigma_x err=1.49e-16, tip disp err=1.90e-16)
[PASS] Test 4: Fixed-fixed thermal bar (sigma_x err=0.00e+00, analytical=-80.5 MPa)
[PASS] Test 5: Mesh convergence verified (tip deflections: ['0.5823mm', '0.5982mm', '0.6024mm', '0.6035mm'])
[PASS] Test 6: Global force equilibrium (Fx balance err=1.65e-14, Fy balance err=3.37e-14)

Ran 6 tests in 0.88s - OK
```

### 2. Launch Interactive Streamlit UI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Parametric Studies Standalone
```bash
python3 -c "
from src.materials import get_default_materials
from src.parametric import run_layup_comparison, run_temperature_sweep, run_fiber_angle_sweep, run_load_increment_to_failure
from src.plotting import plot_parametric_layup_comparison, plot_parametric_deltaT, plot_parametric_angle_sweep, plot_stress_strain_curve

mats = get_default_materials()
carbon = mats['T300_5208_Carbon_Epoxy']

df, _ = run_layup_comparison(carbon, delta_T=50.0, mechanical_tension=100e6, results_dir='results')
plot_parametric_layup_comparison(df, save_path='results/layup_comparison.png')

temp_df = run_temperature_sweep('[0/90]s', carbon, results_dir='results')
plot_parametric_deltaT(temp_df, save_path='results/deltaT_sweep.png')

angle_df = run_fiber_angle_sweep(carbon, results_dir='results')
plot_parametric_angle_sweep(angle_df, save_path='results/angle_sweep.png')

load_df = run_load_increment_to_failure('[0/90]s', carbon, delta_T=50.0, results_dir='results')
plot_stress_strain_curve(load_df, save_path='results/stress_strain_curve.png')
print('Results saved to /results!')
"
```

---

## Governing Physics & Formulations

### 1. Constitutive Relations (Plane Stress)
In principal material coordinates $(1, 2)$:
$$\begin{bmatrix} \sigma_1 \\ \sigma_2 \\ \tau_{12} \end{bmatrix} = \begin{bmatrix} Q_{11} & Q_{12} & 0 \\ Q_{12} & Q_{22} & 0 \\ 0 & 0 & Q_{66} \end{bmatrix} \begin{bmatrix} \epsilon_1^{mech} \\ \epsilon_2^{mech} \\ \gamma_{12}^{mech} \end{bmatrix}$$
where:
$$Q_{11} = \frac{E_1}{1 - \nu_{12}\nu_{21}}, \quad Q_{22} = \frac{E_2}{1 - \nu_{12}\nu_{21}}, \quad Q_{12} = \frac{\nu_{12} E_2}{1 - \nu_{12}\nu_{21}}, \quad Q_{66} = G_{12}$$

### 2. Classical Laminate Theory (CLT)
For arbitrary fiber orientation $\theta$ ($m=\cos\theta, n=\sin\theta$):
$$\bar{\mathbf{Q}} = \mathbf{T}_\epsilon^T \mathbf{Q} \mathbf{T}_\epsilon$$
$$\mathbf{A} = \sum_{k=1}^N \bar{\mathbf{Q}}^{(k)} (z_k - z_{k-1})$$
Mechanical stress in ply $k$:
$$\boldsymbol{\sigma}_{xy}^{(k)} = \bar{\mathbf{Q}}^{(k)} (\boldsymbol{\epsilon}^0 - \boldsymbol{\alpha}_{xy}^{(k)}\Delta T), \quad \boldsymbol{\sigma}_{12}^{(k)} = \mathbf{T}_\sigma(\theta_k) \boldsymbol{\sigma}_{xy}^{(k)}$$

### 3. Thermal-Mechanical Coupling in FEM
Element stiffness and thermal load vectors:
$$\mathbf{K}_e = \int_{-1}^1 \int_{-1}^1 \mathbf{B}^T \mathbf{A} \mathbf{B} \det(\mathbf{J})\,d\xi d\eta$$
$$\mathbf{f}_e^{th} = \int_{-1}^1 \int_{-1}^1 \mathbf{B}^T \mathbf{N}^T(\xi, \eta) \det(\mathbf{J})\,d\xi d\eta$$
where $\mathbf{N}^T(\xi, \eta) = \mathbf{A}\bar{\boldsymbol{\alpha}}_{lam}(T(\xi, \eta) - T_{ref})$.

---

## Engineering Findings & Summary

| Layup | Thickness | $E_x$ [GPa] | $E_y$ [GPa] | $G_{xy}$ [GPa] | $\nu_{xy}$ | $\alpha_x$ [ppm/K] | Max Disp [mm] | Strength Ratio | FPF Mode |
|---|---|---|---|---|---|---|---|---|---|
| `[0]4` | 0.50 mm | **181.0** | 10.3 | 7.17 | 0.280 | **0.02** | 0.060 | 7.79 | In-Plane Shear |
| `[90]4` | 0.50 mm | 10.3 | 181.0 | 7.17 | 0.016 | 22.50 | 1.083 | **0.40 (FAILED)** | Matrix Tension |
| `[0/90]s` | 0.50 mm | 95.99 | 95.99 | 7.17 | 0.030 | 1.52 | 0.112 | **8.19 (SAFEST)** | Fiber Tension |
| `[+45/-45]s` | 0.50 mm | 25.05 | 25.05 | **46.59** | **0.747** | 1.52 | 0.389 | 1.27 | In-Plane Shear |
| `[0/+45/-45/90]s` | 1.00 mm | 69.68 | 69.68 | 26.88 | 0.296 | 1.52 | 0.151 | 5.08 | In-Plane Shear |

- **Axial Stiffness**: `[0]4` yields highest axial stiffness ($E_x = 181\text{ GPa}$) but possesses extremely poor transverse strength ($Y_t = 40\text{ MPa}$).
- **Balanced Strength**: `[0/90]s` provides the highest Strength Ratio ($SR = 8.19$) under combined $100\text{ MPa}$ tension and $\Delta T = 50\text{ K}$.
- **Quasi-Isotropic Behavior**: `[0/+45/-45/90]s` achieves in-plane isotropy ($E_x = E_y = 69.7\text{ GPa}, \nu_{xy} = 0.296$), eliminating thermal distortion.

---

## Model Limitations

1. **Kirchhoff-Love Thin Plate & Plane Stress Assumption**:
   - Out-of-plane transverse normal and shear stresses ($\sigma_z, \tau_{xz}, \tau_{yz}$) are neglected.
   - For thick plates ($H/L > 0.1$), First-Order Shear Deformation Theory (FSDT / Mindlin-Reissner) is required.
2. **Linear Elasticity & First-Ply-Failure**:
   - Small displacement strain kinematics ($\epsilon_{ij} = \frac{1}{2}(u_{i,j} + u_{j,i})$).
   - Elastic response assumed up to First-Ply Failure (FPF). Progressive matrix micro-cracking and post-failure ply discount methods (progressive damage mechanics) are not modeled.
3. **Interlaminar Delamination**:
   - Interlaminar peeling stresses at free edges cannot be resolved using 2D membrane CLT.
4. **Unidirectional Thermal-Structural Coupling**:
   - Weak coupling: temperature changes induce thermal expansion and thermal stresses, but mechanical deformation does not generate heat (negligible thermoelastic dissipation for quasi-static loading).
