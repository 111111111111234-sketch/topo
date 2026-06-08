"""ETPNav adapter compatibility tests.

These tests use a small fake GraphMap so they can run outside the Habitat /
ETPNav environment. The fake exposes the same fields and update/delete methods
that ETPNav's trainer consumes.
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.adapters.etpnav_adapter import ConfTopoETPNavAdapter
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import NodeType
from conftopo.core.instruction_graph import InstructionGraph, SubGoal


class FakeGraphMap:
    def __init__(self):
        self.graph_nx = nx.Graph()
        self.node_pos = {}
        self.node_embeds = {}
        self.node_stepId = {}
        self.ghost_pos = {}
        self.ghost_mean_pos = {}
        self.ghost_embeds = {}
        self.ghost_fronts = {}
        self.ghost_aug_pos = {}
        self.ghost_real_pos = {}
        self.ghost_cnt = 0
        self.shortest_path = {}
        self.shortest_dist = {}

    def update_graph(
        self,
        prev_vp,
        step_id,
        cur_vp,
        cur_pos,
        cur_embeds,
        cand_vp,
        cand_pos,
        cand_embeds,
        cand_real_pos,
    ):
        self.graph_nx.add_node(cur_vp)
        if prev_vp is not None:
            self.graph_nx.add_edge(prev_vp, cur_vp, weight=float(np.linalg.norm(cur_pos - self.node_pos[prev_vp])))
        self.node_pos[cur_vp] = np.asarray(cur_pos, dtype=np.float32)
        self.node_embeds[cur_vp] = np.asarray(cur_embeds, dtype=np.float32)
        self.node_stepId[cur_vp] = int(step_id)

        for cvp, cpos, cemb in zip(cand_vp, cand_pos, cand_embeds):
            ghost_vp = f"g{self.ghost_cnt}"
            self.ghost_cnt += 1
            self.ghost_pos[ghost_vp] = [np.asarray(cpos, dtype=np.float32)]
            self.ghost_mean_pos[ghost_vp] = np.asarray(cpos, dtype=np.float32)
            self.ghost_embeds[ghost_vp] = [np.asarray(cemb, dtype=np.float32), 1]
            self.ghost_fronts[ghost_vp] = [cur_vp]
        self.ghost_aug_pos = dict(self.ghost_mean_pos)
        self.shortest_path = dict(nx.all_pairs_dijkstra_path(self.graph_nx))
        self.shortest_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph_nx))

    def delete_ghost(self, vp):
        self.ghost_pos.pop(vp)
        self.ghost_mean_pos.pop(vp)
        self.ghost_embeds.pop(vp)
        self.ghost_fronts.pop(vp)
        self.ghost_aug_pos = dict(self.ghost_mean_pos)

    def get_node_embeds(self, vp):
        if not str(vp).startswith("g"):
            return self.node_embeds[vp]
        emb, count = self.ghost_embeds[vp]
        return emb / count


def test_adapter_preserves_graphmap_fields_and_methods():
    fake = FakeGraphMap()
    adapter = ConfTopoETPNavAdapter(graph_map=fake)

    cur_emb = np.ones(4, dtype=np.float32)
    cand_emb = [np.zeros(4, dtype=np.float32)]
    adapter.update_graph(
        prev_vp=None,
        step_id=1,
        cur_vp="0",
        cur_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        cur_embeds=cur_emb,
        cand_vp=["0_0"],
        cand_pos=[np.array([1.0, 0.0, 0.0], dtype=np.float32)],
        cand_embeds=cand_emb,
        cand_real_pos=None,
    )

    assert adapter.node_pos is fake.node_pos
    assert adapter.ghost_mean_pos is fake.ghost_mean_pos
    assert adapter.get_node_embeds("0").shape == (4,)
    assert list(adapter.node_pos.keys()) == ["0"]
    assert list(adapter.ghost_mean_pos.keys()) == ["g0"]


def test_adapter_mirrors_graphmap_into_dynamic_topomap():
    adapter = ConfTopoETPNavAdapter(graph_map=FakeGraphMap())
    adapter.update_graph(
        prev_vp=None,
        step_id=1,
        cur_vp="0",
        cur_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        cur_embeds=np.ones(4, dtype=np.float32),
        cand_vp=["0_0", "0_1"],
        cand_pos=[
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ],
        cand_embeds=[np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32) * 2],
        cand_real_pos=None,
    )

    visited = adapter.topo_map.get_nodes_by_type(NodeType.WAYPOINT_VISITED)
    frontiers = adapter.topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER)
    assert len(visited) == 1
    assert len(frontiers) == 2
    assert adapter.topo_map.get_neighbors("etp_0")


def test_adapter_alpha_zero_is_degradation_safe():
    config = ConfTopoConfig()
    config.planning.alpha = 0.0
    adapter = ConfTopoETPNavAdapter(config=config, graph_map=FakeGraphMap())
    adapter.set_instruction_graph(InstructionGraph(goal_type="route", sub_goals=[
        SubGoal(id=0, action="go_forward", landmark="door"),
    ]))
    adapter.update_graph(
        prev_vp=None,
        step_id=1,
        cur_vp="0",
        cur_pos=np.zeros(3, dtype=np.float32),
        cur_embeds=np.ones(4, dtype=np.float32),
        cand_vp=["0_0"],
        cand_pos=[np.ones(3, dtype=np.float32)],
        cand_embeds=[np.ones(4, dtype=np.float32)],
        cand_real_pos=None,
    )

    bias = adapter.get_semantic_bias([None, "0", "g0"], np.zeros(3, dtype=np.float32))
    assert np.allclose(bias, 0.0)


def run_all():
    test_adapter_preserves_graphmap_fields_and_methods()
    test_adapter_mirrors_graphmap_into_dynamic_topomap()
    test_adapter_alpha_zero_is_degradation_safe()
    print("[PASS] ETPNav adapter compatibility")


if __name__ == "__main__":
    run_all()

