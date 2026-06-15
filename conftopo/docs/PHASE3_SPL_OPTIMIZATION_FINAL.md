# Phase 3 SPL Optimization: Final Report

## Summary

SPL improved from 0.0172 to 0.6333 (37x) across 4 rounds of fixes.
Core issue: detail_score locked by frequent heavy perception prevented folding.
SR: 25% -> 100% (4/4). Heavy calls: 320 -> 46. Object merges: 473 -> 13.

## Changes

### P1: Object Anchor Retention
- _mark_node_folded_anchor() / _mark_node_active_detail() in dynamic_topo_map.py
- get_object_anchor() retrieves anchor waypoint + room info
- room_level_min_distance 6.0 -> 4.5, staleness decay for mid-range objects
- _candidate_skip_reason allows folded goal objects through
- _target_output_for_node redirects to anchor waypoint

### P2: Room Semantic Summary
- _absorb_object_into_room_summary() with count/confidence/representative objects
- _format_room_summary_text() for debug
- Called from _update_room_summary()

### P3: Local Re-grounding
- 6 new methods + state machine in goat_agent.py
- scanning (8 rotations) -> searching (30 steps) -> failed
- Heavy perception bypasses cooldown during reground
- _try_start_regrounding triggers when agent within 0.8m of anchor

### P4: Global Graph Planning (Dijkstra)
- shortest_path() with edge_type parameter
- GlobalGraphPlanner in pathfinder_executor.py
- _resolve_navigable_target() resolves any node to waypoint
- Disabled by default (global_graph_enabled: bool = False)

### Bug fixes
1. detail_score recency lock: room_level_min_distance 6.0->4.5 + staleness decay
2. target_relevance permanent: reset on set_new_goal()
3. JSON ndarray: .copy() -> .tolist()
4. anchor_waypoint_position missing: added to extras
5. folded_nodes count: folded_count -> semantic_anchor_count

## Results

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| SR | 25% (1/4) | 100% (4/4) | +75% |
| avg SPL | 0.0172 | 0.6333 | 37x |
| Task 3 SPL | failed | 1.0 (perfect) | +inf |
| Heavy calls | 320 | 46 | -85% |
| Object merges | 473 | 13 | -97% |
| Memory preserved | 1/4 | 4/4 | +75% |

## Experiment Commands

Standard multi-goal eval:
  cd /workspace/tangyx7@xiaopeng.com &&
  python scripts/run_goat_multigoal_acceptance.py
    --split val_seen --scene GLAQ4DNUx5U --episode-index 0
    --max-goals 4 --steps-per-goal 80
    ... --heavy-enabled ... --record-topo

Check folding stats:
  python3 -c "import json; t=json.load(open('TRACE')); s=t['steps'][-1]; m=s['memory']; g=m['granularity_debug']; print('folded:',m['folded_nodes'],'anchors:',m['semantic_anchors'],'far:',g['far_object_candidates'],'fold_marks:',g['folded_anchor_marks'])"

## Files Modified

config.py: 1 line (+global_graph_enabled)
dynamic_topo_map.py: +149 lines (anchor, summary, staleness, edge_type)
goat_agent.py: +257 lines (skip_reason, target_output, reground state machine, resolve_target, plan integration)
pathfinder_executor.py: +13 lines (GlobalGraphPlanner)


## P5: Edge Description

### 动机
让 topo graph 的边可解释，支持 debug trace 可视化、LLM route reasoning、拟人导航表达。

### 改动
只改 ，新增 8 个组件：

| 组件 | 类型 | 行 | 说明 |
|------|------|-----|------|
|  | import | 8 | |
|  | 模块级函数 | ~85 | 角度归一化 |
|  | 模块级函数 | ~89 | 8-direction 标签 |
|  | 方法 | ~316 | NAVIGABLE 边结构化 |
|  | 方法 | ~345 | portal 类型推断 |
|  | 方法 | ~361 | ADJACENT_TO 边结构化 |
|  | 模块级函数 | EOF | 路线描述 |
|  | 模块级函数 | EOF | portal 描述 |

### NAVIGABLE 边结构化



### ADJACENT_TO (portal) 边结构化



### 路线描述函数



### 设计约束

- 不影响 planner scoring / Dijkstra cost
- 不影响序列化 (所有值都是 list / str / float / int)
-  和  用 dict 适配无向图双向语义
-  使用世界坐标方向 (无 agent heading)，避免无向边语义混乱


## P5: Edge Description

### 动机
让 topo graph 的边可解释，支持 debug trace 可视化、LLM route reasoning、拟人导航表达。

### 改动
只改 dynamic_topo_map.py，新增 8 个组件：

| 组件 | 类型 | 行 | 说明 |
|---|---|---|------|
| import math | import | 8 | |
| _normalize_angle() | 模块级函数 | ~85 | 角度归一化 |
| _heading_to_direction_label() | 模块级函数 | ~89 | 8-direction 标签 |
| _enrich_navigable_edge() | 方法 | ~316 | NAVIGABLE 边结构化 |
| _infer_passage_type() | 方法 | ~345 | portal 类型推断 |
| _enrich_adjacent_edge() | 方法 | ~361 | ADJACENT_TO 边结构化 |
| format_edge_description() | 模块级函数 | EOF | 路线描述 |
| format_portal_description() | 模块级函数 | EOF | portal 描述 |
