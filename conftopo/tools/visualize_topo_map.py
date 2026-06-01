"""Visualize DynamicTopoMap as a graph figure."""

import numpy as np
from typing import Optional

from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, EdgeType


# Color scheme for node types
NODE_COLORS = {
    NodeType.WAYPOINT_VISITED: "#2196F3",   # blue
    NodeType.WAYPOINT_FRONTIER: "#FF9800",  # orange
    NodeType.LANDMARK: "#9C27B0",           # purple
    NodeType.OBJECT: "#4CAF50",             # green
    NodeType.ROOM: "#F44336",              # red
}

NODE_SHAPES = {
    NodeType.WAYPOINT_VISITED: "o",
    NodeType.WAYPOINT_FRONTIER: "^",
    NodeType.LANDMARK: "s",
    NodeType.OBJECT: "D",
    NodeType.ROOM: "H",
}


def visualize_topo_map(
    topo_map: DynamicTopoMap,
    output_path: Optional[str] = None,
    agent_pos: Optional[np.ndarray] = None,
    title: str = "DynamicTopoMap",
    figsize: tuple = (12, 10),
    show_labels: bool = True,
    show_confidence: bool = True,
):
    """Visualize the topological map.

    Args:
        topo_map: the map to visualize
        output_path: if provided, save figure to this path
        agent_pos: current agent position (shown as star)
        title: figure title
        figsize: figure size
        show_labels: show node labels
        show_confidence: scale node size by confidence
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw edges
    for u, v, data in topo_map.graph.edges(data=True):
        node_u = topo_map.get_node(u)
        node_v = topo_map.get_node(v)
        if node_u is None or node_v is None:
            continue
        edge_type = data.get("edge_type", "navigable")
        # 实虚线
        linestyle = "-" if edge_type == EdgeType.NAVIGABLE.value else "--"
        alpha = 0.6 if edge_type == EdgeType.NAVIGABLE.value else 0.3
        ax.plot(
            [node_u.position[0], node_v.position[0]],
            [node_u.position[1], node_v.position[1]],
            color="gray", linestyle=linestyle, alpha=alpha, linewidth=0.8,
        )

    # Draw nodes by type
    for node_type in NodeType:
        nodes = topo_map.get_nodes_by_type(node_type)
        if not nodes:
            continue
        xs = [n.position[0] for n in nodes]
        ys = [n.position[1] for n in nodes]
        if show_confidence:
            sizes = [max(30, n.confidence * 200) for n in nodes]
        else:
            sizes = [80] * len(nodes)

        ax.scatter(
            xs, ys,
            c=NODE_COLORS[node_type],
            s=sizes,
            marker=NODE_SHAPES[node_type],
            label=node_type.value,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )

        if show_labels:
            for node in nodes:
                if node.label:
                    ax.annotate(
                        node.label,
                        (node.position[0], node.position[1]),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points",
                    )

    # Draw agent position
    if agent_pos is not None:
        ax.scatter(
            agent_pos[0], agent_pos[1],
            c="yellow", s=300, marker="*",
            edgecolors="black", linewidth=1.5,
            label="Agent", zorder=10,
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def print_map_stats(topo_map: DynamicTopoMap):
    """Print summary statistics of the map."""
    print(f"=== DynamicTopoMap Stats (step {topo_map.current_step}) ===")
    print(f"  Total nodes: {topo_map.num_nodes}")
    for nt in NodeType:
        nodes = topo_map.get_nodes_by_type(nt)
        if nodes:
            confs = [n.confidence for n in nodes]
            print(f"  {nt.value}: {len(nodes)} (conf: {np.mean(confs):.2f} ± {np.std(confs):.2f})")
    print(f"  Total edges: {topo_map.graph.number_of_edges()}")
    edge_types = {}
    for _, _, data in topo_map.graph.edges(data=True):
        et = data.get("edge_type", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1
    for et, count in edge_types.items():
        print(f"    {et}: {count}")
