import numpy as np
import plotly.graph_objects as go


def patch_traces(
    patch_nodes: np.ndarray,
    phi: np.ndarray,
    bitstring: str,
    patch_id: str,
    show_phi: bool = False,
):
    """
    Returns Plotly traces (NOT a figure).
    """

    patch_nodes = np.asarray(patch_nodes)
    phi = np.asarray(phi)
    selected = np.array([b == "1" for b in bitstring])

    traces = []

    # --- all nodes (background) ---
    traces.append(
        go.Scatter(
            x=patch_nodes[:, 0],
            y=patch_nodes[:, 1],
            mode="markers",
            marker=dict(
                size=2,
                color=phi if show_phi else "black",
                colorscale="Viridis",
                showscale=show_phi,
            ),
            name=f"{patch_id} nodes",
            legendgroup=patch_id,
            showlegend=False,
        )
    )

    # --- selected nodes ---
    if selected.any():
        traces.append(
            go.Scatter(
                x=patch_nodes[selected, 0],
                y=patch_nodes[selected, 1],
                mode="markers",
                marker=dict(
                    size=3,
                    color="red",
                    symbol="x",
                ),
                name=f"{patch_id} selected",
                legendgroup=patch_id,
                showlegend=False,
            )
        )

    return traces


def combined_figure(all_traces, title="All patch selections"):
    fig = go.Figure(data=all_traces)

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )

    return fig


def single_patch_figure(
    patch_nodes, phi, bitstring, patch_id
):
    fig = go.Figure(
        data=patch_traces(
            patch_nodes,
            phi,
            bitstring,
            patch_id,
            show_phi=True,
        )
    )

    fig.update_layout(
        title=f"Patch {patch_id}",
        template="plotly_white",
    )

    return fig
