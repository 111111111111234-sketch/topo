# Phase 3 SPL Optimization Report

> Phase 3 dynamic semantic topo memory 的 SPL 优化改动记录与效果评估。

## 概要

针对 Phase 3 多目标连续导航中 SPL 过低 (0.0172) 的问题，做了 3 轮迭代修复。核心问题是：**detail_score 被频繁的 heavy perception 锁定在 0.65+，导致 adaptive_granularity 永不压缩、永不 folding、永不创建 semantic anchor**。修复后折叠-锚点-regrounding 链路生效，SPL 提升 37 倍。

---

## 改动总览

### 文件级修改

| 文件 | 行数变化 | 改动数 |
|------|---------|-------|
| `conftopo/config.py` | 122 (不变) | 1 |
| `conftopo/core/dynamic_topo_map.py` | 2073 → 2222 (+149) | 10 |
| `conftopo/agents/goat_agent.py` | 1304 → 1553 (+249) | 12 |
| `conftopo/navigation/pathfinder_executor.py` | 215 (不变) | 0 |

---

### 改动明细

#### config.py

| 改动 | 行 | 旧值 | 新值 | 目的 |
|------|-----|------|------|------|
| `room_level_min_distance` | 32 | 6.0 | 4.5 | 降低远距离压缩触发阈值 |

#### dynamic_topo_map.py

| 改动 | 行 | 说明 |
|------|-----|------|
| `_last_granularity_debug` 初始化 | 156 | debug 追踪 |
| `_mark_node_active_detail()` | 1755 | 展开节点 (不重置 is_semantic_anchor) |
| `_mark_node_folded_anchor()` | 1761 | 折叠节点 + 保存 anchor 信息 |
| 默认 `room_level_min_distance` | 145 | 8.0 → 4.5 |
| Staleness decay | 882-889 | 3+步未检测的 mid-range 对象自动降 detail_score |
| `.copy()` → `.tolist()` | 1774, 1781 | JSON 序列化修复 |
| `_absorb_object_into_room_summary()` | ~1923 | room summary 带 label 计数、置信度、代表作 |
| `_format_room_summary_text()` | ~1965 | 模板化 summary 文本 |
| `get_object_anchor()` | 209 | 检索折叠节点 anchor 信息 |
| `_normalize_semantic_label()` | ~1917 | label 归一化 |

#### goat_agent.py

| 改动 | 行 | 说明 |
|------|-----|------|
| target_relevance 重置 in `set_new_goal()` | ~219 | 每次切换 goal 重置所有 nodes |
| `_candidate_skip_reason()` | 1054 | folded goal 对象不跳过 |
| `_target_output_for_node()` | 461 | is_semantic_anchor → anchor waypoint 重定向 |
| `_try_start_regrounding()` | 412 | anchor 到达触发 scanning |
| `_reground_scan_action()` | 441 | 旋转扫描 |
| `_plan_to_reground_target()` | 1208 | reground 目标导航 |
| `_fail_regrounding()` | 1229 | 超时/失败处理 |
| `_plan_local_reground_search()` | 1237 | 有限 frontier 搜索 |
| `_plan_regrounding()` | 1277 | scanning→searching 状态机 |
| `plan()` | 1300 | 优先检查 _plan_regrounding() |
| `act()` | 1418 | 优先处理 local_reground_scan 和 _try_start_regrounding() |
| `_should_run_heavy_perception()` | 834 | reground 时绕过 cooldown |
| `_add_heavy_object_nodes()` | 961 | reground 时设 _target_object_detected_this_scan |
| `memory_stats` | ~1530 | 新增统计字段 |
| folded_nodes 计数修复 | 1547 | folded_count → semantic_anchor_count |

---

## 修复的 Bug

### Bug 1: detail_score 被 recency 锁死 (致命)

**表现**: adaptive_granularity() 永不压缩

**原因**: _semantic_detail_score() 中 recency_score 权重 0.20，heavy perception 每步刷新 last_seen_step → recency=1.0 → detail_score ≈ 0.73 → compression 永不触发

**修复**:
- room_level_min_distance 6.0 → 4.5, >4.5m 对象无条件压缩
- staleness decay: 3+步未检测的 mid-range 对象自动降分

### Bug 2: target_relevance 永驻阻止 folding

**表现**: _update_node_visibility 永远展开曾是目标的 objects

**原因**: _merge_object_observation 用 max(old, new) 合并 target_relevance，一旦设过 1.0 永不归零

**修复**: set_new_goal() 重置所有 nodes target_relevance = 0

### Bug 3: JSON 序列化 numpy array

**表现**: TypeError: ndarray not JSON serializable

**原因**: _mark_node_folded_anchor 用 .copy() 存 ndarray 到 attributes

**修复**: .copy() → .tolist(), get_object_anchor 用 np.array() 重构

### Bug 4: anchor_waypoint_position 未传递

**表现**: regrounding 永不触发

**原因**: _target_output_for_node() 的 extras dict 缺 anchor_waypoint_position

**修复**: extras.update 加入 `"anchor_waypoint_position"`

### Bug 5: folded_nodes 计数不准

**表现**: folded_anchor_marks=2 但 folded_nodes=0

**原因**: _update_node_visibility 展开节点时重置 folded=False 但不重置 is_semantic_anchor

**修复**: folded_nodes 用 semantic_anchor_count 替代 folded_count

---

## 效果对比

### 核心指标

| 指标 | round2_v5 (旧) | round4 (修复后) | 变化 |
|------|----------------|-----------------|------|
| **Success Rate** | 1/4 (25%) | **4/4 (100%)** | +75% |
| **avg SPL** | 0.0172 | **0.6333** | **37x** |
| Task 0 SPL (plush toy, 新) | 0.0688 | 0.3202 | +365% |
| Task 1 SPL (wardrobe, 新) | failed | **0.7345** | ✓ |
| Task 2 SPL (wardrobe, 复用) | failed | **0.4786** | ✓ |
| Task 3 SPL (plush toy, 复用) | failed | **1.0** (完美) | ✓ |

### 行为指标

| 指标 | v5 (旧) | round4 | 变化 |
|------|---------|--------|------|
| Heavy perception calls | 320 | **46** | **-85%** |
| Object merges | 473 | **13** | **-97%** |
| Memory preserved | 1/4 | **4/4** | 全部保持 |
| Collision count | 0 | 2 | 可忽略 |

### 中间过程

```
round2_v5 (基线)
  SR=25% avg_SPL=0.0172  heavy_calls=320
    └─ room_level_min_distance 6.0→4.5 + staleness decay
    └─ _mark_node_folded_anchor + anchor retention
round3_multigoal_fix
  SR=100% avg_SPL=0.4254  heavy_calls=47
    └─ target_relevance 归零修复
    └─ .tolist() JSON fix
round4_multigoal_fix (当前)
  SR=100% avg_SPL=0.6333  heavy_calls=46
```

## 技术架构

### 折叠-锚点-重新定位链路

```
agent 离开 object > 4.5m
→ adaptive_granularity: dist > room_level_min_distance
→ 压缩 + _mark_node_folded_anchor
→ folded=True, is_semantic_anchor=True
→ _update_node_visibility: 条件展开 (folded=False), anchor 保留

新 goal: "find plush toy"
→ _candidate_skip_reason: is_semantic_anchor + _is_goal_object_node → 不跳过
→ plan() 选中
→ _target_output_for_node: redirect 到 anchor waypoint
→ navigate to anchor waypoint (目标 SPL 提升来源)

到达 anchor waypoint (dist < 0.8m)
→ _try_start_regrounding: _reground_state = "scanning"
→ 每步 heavy perception (reground 绕过 cooldown)
→ 检测到目标 → _target_object_detected_this_scan = True
→ scanning → searching → navigate 到精确位置
```

### 状态机

```
idle → plan 选中 folded goal object + navigate to anchor
     → anchor_reached: scanning (rotate 8x)
         → 检测到目标 → searching → navigate → idle
         → 未检测到 → searching → frontier 搜索 (max 30 steps)
             → 找到候选 → navigate → idle
             → 超时 → failed → idle (禁止重试同一 anchor)
```

---

## 后续建议

1. **多场景验证**：当前只在 scene=GLAQ4DNUx5U 上测试，需扩展到更多场景
2. **Ablation**：验证每个改动的独立贡献 (target_relevance 归零 vs room_level_min_distance vs staleness decay)
3. **Global graph planning**：当前 plan() 仍用贪婪打分，可接入 Dijkstra/A* 全局路径
4. **Registration**：建议做原始数据记录以便后续分析和论文

