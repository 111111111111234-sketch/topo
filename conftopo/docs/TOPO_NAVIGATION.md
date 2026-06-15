# TopoMap Navigation Logic - ConfTopo-GOAT

> 本文档说明 ConfTopo-GOAT Agent 的完整导航拓扑逻辑，包括感知、记忆、规划、决策的循环。
>
> Multigoal harness 执行流程、goal 切换、SR/SPL 评估见 [MULTIGOAL_NAVIGATION.md](./MULTIGOAL_NAVIGATION.md)。

---

## 1. Agent Step Loop

每步执行 `observe - update_memory - plan - act` 循环：

```
step()
  |-- observe(obs)         # 处理观测: 位置/朝向/CLIP embedding
  |-- update_memory()      # 更新拓扑图: waypoint/semantic/frontier
  |-- plan()               # 选择下一个导航目标
  +-- act(plan_output)     # 输出 target_position -> 底层控制器
```

### 1.1 observe()

**输入**: 观测字典 `{rgb, rgb_embed, position, heading}`

关键行为:
- 位置从世界坐标转为 episode-start 相对坐标
- 每步运行 CLIP light perception
- `_prev_position` 在上一步保存，用于 frontier 生成

### 1.2 update_memory()

```
def update_memory(self):
    self._goal_local_step += 1
    cur_vp = self._add_visited_waypoint()              # 1) 添加已访问 waypoint
    self._consume_reached_frontiers(cur_vp)            # 消耗已到达的 frontier
    room_label = self._add_semantic_nodes(cur_vp)      # 2) 添加语义节点
    self._add_heavy_object_nodes(cur_vp, room_label)  # 2b) GroundingDINO 重检测
    self._generate_frontiers(cur_vp)                   # 3) 生成新的 frontier
    self.topo_map.decay_all_confidences()              # 4) 记忆维护
    self.topo_map.merge_nearby_nodes()
    self.topo_map.adaptive_granularity()
    self.topo_map.prune_low_confidence()
```

---

## 2. TopoMap 结构

文件: `conftopo/core/dynamic_topo_map.py`

### 2.1 节点类型

| 节点类型 | 作用 |
|---------|------|
| WAYPOINT_VISITED | Agent 经过的位置，间距 ~0.5m |
| WAYPOINT_FRONTIER | 未探索方向的候选位置 |
| WAYPOINT_CANDIDATE | 消耗后的 frontier 标记 |
| OBJECT | GroundingDINO/CLIP 检测到的物体 |
| LANDMARK | 导航锚点(门、走廊等结构性地标) |
| ROOM | 房间/区域摘要节点 |

### 2.2 边类型

| 边类型 | 作用 |
|-------|------|
| NAVIGABLE | waypoint-to-waypoint 可通行路径 |
| OBSERVED_AT | waypoint-to-object 观测关系 |
| BELONGS_TO | waypoint/object-to-room 归属关系 |

### 2.3 DynamicTopoMap

```
class DynamicTopoMap:
    _nodes: Dict[str, SemanticNode]
    _edges: Dict[str, dict]       # key="src->tgt", value={edge_type, weight}
    current_step: int
```

---

## 3. 感知系统

### 3.1 Light Perception (CLIP)

每步运行，输出 room/object/landmark 的 CLIP 相似度评分。

阈值: object=0.05 (Phase2 统计), room=0.20, landmark=0.22

### 3.2 Heavy Perception (GroundingDINO)

按需触发(heavy_interval=7，frontier/低置信/靠近 summary 时触发)。

**默认检测词汇**:
`rack`, `chair`, `table`, `door`, `sofa`, `bed`, `sink`, `toilet`, `cabinet`, `fridge`, `tv`, `plant`

---

## 4. Frontier 生成

文件: `goat_agent.py - _generate_frontiers()`

```
条件: displacement > 0.5m (min_move_for_frontier)

4 方向 (0, 90, 180, -270 deg) 距离 2.5m 生成 WAYPOINT_FRONTIER
已访问/frontier 1.5m 内跳过 (merge_radius)
```

---

## 5. 置信度系统

文件: `conftopo/core/confidence.py`

```
score = max(detection, base/0.9) * 0.995^staleness - penalty
penalty = 0.05*redundancy + 0.1*conflict
base = 0.3*detection + 0.25*multi_view + 0.2*relevance + 0.15*room_prior
```

---

## 6. 记忆维护

- **自适应粒度**: object(近) -> landmark(中) -> room_level(远)
- **剪枝**: 远距离低置信 object(conf<0.18, dist>10m) 删除
- **Waypoint 合并**: 同类型间距 < 1.0m 合并

---

## 7. 两阶段规划 (Two-Stage Planner)

### Stage 1: 选择结构锚点

从结构层节点(ROOM + 结构性 LANDMARK)中评分最高的作为锚点。
阈值 0.05，未命中则回退到单阶段。距 agent 最近的锚点 +0.1 距离梯度奖励。

### Stage 2: 评分导航候选

候选: primary (frontier/object/room_summary) + fallback (visited waypoint)
不选择条件: consumed/blocked/anchored_skip(far_object)

评分: `compute_semantic_bias()` + `structure_anchor_bonus()`

Sticky target: 保持目标直到到达 0.75m 或 5 步无进展。

---

## 8. 评分系统

### 通用评分公式

```
score = object_match(0.55*sim) + frontier_bonus(0.1)
       - distance_penalty(0.08*dist/20) - visit_penalty(0.1)
       + confidence_bonus(0.1*conf)
```

### 物体目标评分(GOAT)

| 条件 | 加分 |
|------|:----:|
| OBJECT 匹配目标 | 0.55 * cos_sim |
| ROOM 匹配 room_prior | 0.3 |
| Room summary 包含目标 | 0.25 |
| waypoint 附近有匹配物体(sim>0.5) | 0.2 * sim |
| 附近 landmark 匹配 | 0.2 * max_sim |

### 结构锚点奖励

```
scores[i] += bonus(0.25) * (1.0 - dist/radius(6.0))
if room_id == structure_target_id: scores[i] += bonus(0.25)
```

---

## 9. 配置参数

| 参数 | 默认值 | 作用 |
|------|:------:|------|
| heavy_interval | 7 | GroundingDINO 触发间隔(步) |
| object_threshold | 0.05 | CLIP 物体检测阈值(Phase2) |
| room_threshold | 0.20 | CLIP 房间识别阈值 |
| two_stage_enabled | True | 两阶段规划开关 |
| structure_anchor_bonus | 0.25 | 结构锚点奖励值 |
| structure_anchor_radius | 6.0m | 锚点作用半径 |
| sticky_reach_radius | 0.75m | 目标到达判定距离 |
| sticky_release_no_progress | 5 | 无进展释放步数 |
| frontier_step_size | 2.5m | frontier 生成距离 |
| min_move_for_frontier | 0.5m | frontier 生成最小位移 |

---

## 10. 当前性能

14 场景 val_seen, 500 步/目标, 单 episode

| 指标 | 值 | 说明 |
|------|:---:|------|
| SR | 85/112 (75.9%) | 目标成功率 |
| avg SPL | 0.069 | 路径效率(低=效率低) |
| heavy_perception | ~143 calls | GroundingDINO 调用次数 |
| object_nodes | ~50 | 语义物体节点数 |

**已修复瓶颈**(Phase 3.8):
- bbox 位置估计: 从固定 2m 改为基于 bbox 大小的自适应距离(0.5~2.0m)
- should_stop: Heavy 匹配加 warmup 门槛(需 step>5 且 travel>0.5m)
- 复合目标: "oven and stove" 自动拆分为 {"oven","stove"} 匹配任一
- 评分权重: frontier 0.2→0.1, object_match 0.4→0.55, distance_penalty 0.05→0.08