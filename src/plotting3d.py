"""
Interactive Plotly 3D views for the live simulator:
- exploded ply-by-ply layer stack (each ply colored by its own field)
- bent plate surface (out-of-plane deflection, ANSYS-style contour)
- animated transient heating of the deformed plate
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates["plate"] = go.layout.Template(layout=dict(
    font=dict(family="Archivo, system-ui, sans-serif", color="#14222E"),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF"))
pio.templates.default = "plotly_white+plate"
from src.mesh import QuadMesh
from src.plotting import q4_to_triangles


def elem_to_nodal(mesh: QuadMesh, elem_vals: np.ndarray) -> np.ndarray:
    """Average element values (Ne, ...) onto nodes (Nn, ...)."""
    out = np.zeros((mesh.num_nodes,) + elem_vals.shape[1:])
    cnt = np.zeros(mesh.num_nodes)
    for c in range(4):
        np.add.at(out, mesh.elements[:, c], elem_vals)
        np.add.at(cnt, mesh.elements[:, c], 1.0)
    return out / cnt.reshape((-1,) + (1,) * (elem_vals.ndim - 1))


def _scene(aspect_z: float = 0.6, eye: tuple = (-1.45, -2.0, 1.35)) -> dict:
    return dict(
        xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="",
        aspectmode="manual", aspectratio=dict(x=2, y=1, z=aspect_z),
        camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])),
    )


def layer_stack_figure(mesh: QuadMesh, ply_nodal: np.ndarray, ply_labels: list,
                       displacements: np.ndarray = None, scale: float = 0.0,
                       cbar_title: str = "", colorscale: str = "Jet") -> go.Figure:
    """ply_nodal: (num_plies, num_nodes). Ply 1 at the bottom of the stack."""
    xy = mesh.nodes + (scale * displacements if displacements is not None else 0.0)
    x, y = xy[:, 0] * 1e3, xy[:, 1] * 1e3
    tri = q4_to_triangles(mesh.elements)
    gap = 0.35 * max(mesh.Lx, mesh.Ly) * 1e3
    cmin, cmax = float(ply_nodal.min()), float(ply_nodal.max())
    if cmax - cmin < 1e-6 * max(1.0, abs(cmin)):
        cmin, cmax = cmin - 1.0, cmax + 1.0

    fig = go.Figure()
    for k, (vals, label) in enumerate(zip(ply_nodal, ply_labels)):
        z = np.full_like(x, k * gap)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            intensity=vals, colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=(k == 0), colorbar=dict(title=cbar_title),
            name=label, hovertemplate=f"{label}<br>x=%{{x:.1f}} mm<br>y=%{{y:.1f}} mm<br>value=%{{intensity:.3g}}<extra></extra>",
            flatshading=True, lighting=dict(ambient=0.9, diffuse=0.2),
        ))
        fig.add_trace(go.Scatter3d(
            x=[x.min() - 0.05 * gap], y=[y.max()], z=[k * gap], mode="text",
            text=[label], textposition="middle left", showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(scene=_scene(1.2, eye=(-1.7, -2.4, 1.5)), height=650, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_scenes(zaxis_visible=False)
    return fig


def bending_surface_figure(X: np.ndarray, Y: np.ndarray, W: np.ndarray, color: np.ndarray,
                           scale: float, cbar_title: str, colorscale: str = "Jet") -> go.Figure:
    Xmm, Ymm = X * 1e3, Y * 1e3
    fig = go.Figure(go.Surface(
        x=Xmm, y=Ymm, z=W * 1e3 * scale, surfacecolor=color, colorscale=colorscale,
        colorbar=dict(title=cbar_title),
        customdata=W * 1e3,
        hovertemplate="x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>w=%{customdata:.4f} mm<br>value=%{surfacecolor:.3g}<extra></extra>",
    ))
    # Undeformed outline
    xo = [Xmm.min(), Xmm.max(), Xmm.max(), Xmm.min(), Xmm.min()]
    yo = [Ymm.min(), Ymm.min(), Ymm.max(), Ymm.max(), Ymm.min()]
    fig.add_trace(go.Scatter3d(x=xo, y=yo, z=[0] * 5, mode="lines",
                               line=dict(color="black", width=4), name="Undeformed", hoverinfo="skip"))
    fig.update_layout(scene=_scene(0.5), height=600, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_scenes(zaxis=dict(title="", showticklabels=False))
    return fig


def transient_animation_figure(mesh: QuadMesh, times: np.ndarray, T_frames: np.ndarray,
                               disp_frames: np.ndarray, scale: float) -> go.Figure:
    """T_frames: (F, Nn) in °C; disp_frames: (F, Nn, 2) in m. Plate drawn with height ∝ |u|."""
    tri = q4_to_triangles(mesh.elements)
    tmin, tmax = float(T_frames.min()), float(T_frames.max())
    if tmin == tmax:
        tmax = tmin + 1e-9

    def mesh_trace(f):
        xy = (mesh.nodes + scale * disp_frames[f]) * 1e3
        return go.Mesh3d(x=xy[:, 0], y=xy[:, 1], z=np.zeros(mesh.num_nodes),
                         i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                         intensity=T_frames[f], colorscale="Jet", cmin=tmin, cmax=tmax,
                         colorbar=dict(title="T [°C]"), flatshading=True,
                         lighting=dict(ambient=1.0, diffuse=0.0),
                         hovertemplate="T=%{intensity:.1f} °C<extra></extra>")

    labels = [f"t = {t:.0f} s" for t in times]
    frames = [go.Frame(data=[mesh_trace(f)], name=labels[f]) for f in range(len(times))]
    last = len(times) - 1
    fig = go.Figure(data=[mesh_trace(last)], frames=frames)  # open on the heated state, not the cold start

    pad = 0.1 * max(mesh.Lx, mesh.Ly) * 1e3
    all_xy = (mesh.nodes[None] + scale * disp_frames) * 1e3
    fig.update_layout(
        height=550, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis=dict(range=[all_xy[..., 0].min() - pad, all_xy[..., 0].max() + pad], title="x [mm]"),
                   yaxis=dict(range=[all_xy[..., 1].min() - pad, all_xy[..., 1].max() + pad], title="y [mm]"),
                   zaxis=dict(visible=False, range=[-1, 1]),
                   aspectmode="manual", aspectratio=dict(x=2, y=1, z=0.2),
                   camera=dict(eye=dict(x=-0.5, y=-1.3, z=1.2))),
        updatemenus=[dict(type="buttons", x=0.0, y=0.05, showactive=False, buttons=[
            dict(label="▶ Play", method="animate",
                 args=[labels, dict(frame=dict(duration=120, redraw=True), mode="immediate", transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ])],
        sliders=[dict(active=last, x=0.12, len=0.85, y=0.05, currentvalue=dict(prefix=""),
                      steps=[dict(label=l, method="animate",
                                  args=[[l], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for l in labels])],
    )
    return fig

