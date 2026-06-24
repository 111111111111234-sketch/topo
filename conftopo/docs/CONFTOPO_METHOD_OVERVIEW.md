# ConfTopo 方法思路整理

> 项目级认知叙事与总体定位见
> [CONFTOPO_DESIGN_SUMMARY.md](CONFTOPO_DESIGN_SUMMARY.md)。

## 1. 核心问题

GOAT-Bench 多目标导航的难点不是单次目标识别，而是 agent 需要在一个 episode 中连续完成多个目标：

```text
看见环境 -> 记住结构 -> 复用记忆 -> 根据当前目标重新规划
```

传统端到端 policy 或单帧 VLM 决策容易遇到三个问题：

1. **短期感知强，长期记忆弱**：当前帧能识别物体，但走远后难以稳定复用。
2. **语义和空间割裂**：知道目标是什么，但不知道它和房间、门、走廊、已访问路径的关系。
3. **多目标切换成本高**：每个新目标都像重新探索，无法充分利用前一个目标过程中积累的环境知识。

ConfTopo 的核心思想是：  

> 用视觉语义感知构建一个动态拓扑记忆图，再在图上做任务驱动的结构化规划。

也就是把导航问题从“每一步直接预测动作”改成：

```text
感知当前视野
-> 写入长期拓扑图
-> 在图中选择下一个语义/空间目标
-> 交给底层控制器执行
```

## 2. 方法总览

ConfTopo 可以概括为四个能力：

```text
会看：Perception
会记：DynamicTopoMap
会想：Task-driven Graph Planning
会走：Navigator / PathfinderExecutor
```

整体闭环：

```text
RGB / pose / goal
    |
    v
PerceptionReport
    |
    v
DynamicTopoMap
    |
    v
Planner
    |
    v
target_position / stop / local scan
    |
    v
Navigator
```

在当前实现中，主要入口仍是：

```text
conftopo/agents/goat_agent.py
```

内部执行：

```text
observe -> update_memory -> plan -> act
```

后续架构演进会逐步拆成：

```text
GoalAgent
PerceptionAgent
MemoryAgent
PlannerAgent
Navigator
```

并通过 Blackboard 共享状态。

## 3. 感知：从帧级语义到 PerceptionReport

ConfTopo 不直接让 planner 依赖某个具体模型输出，而是把感知结果统一成：

```text
PerceptionReport
```

它表示当前视角下的结构化场景理解：

```text
room_label / room_confidence
goal_scores
landmark_scores
objects
scene_summary
goal_visible
portals
uncertainty
visual_embed
```

当前推荐感知策略是双轨制：

```text
每步轻量感知：
CLIP visual embedding -> room / goal / landmark score

触发式重感知：
GroundingDINO 或 VLM -> object / room / portal / goal visibility
```

当前代码里，这条分工已经进一步收紧为：

```text
CLIP / LightPerceiver
-> hypothesis / proposal / context / frontier_value

VLM / HeavyPerceiver
-> verified object observation -> Object Memory
```

VLM 不每步调用，而是在关键时刻触发：

```text
goal switch
low confidence
near goal
stuck
local regrounding
periodic refresh
stop confirmation
```

这样做的目的是：既利用 VLM 的强语义理解，又避免在线导航中每步调用大模型造成不可接受的延迟。

### 3.1 当前感知边界

`PerceptionReport` 仍然是统一接口，但实际上分成两层：

```text
light report:
  room_scores / goal_scores / landmark_scores / best_goal_sim

full report:
  objects / scene_summary / goal_visible / stop_candidate / attributes ...
```

当前实现中：

- `light report` 主要来自 CLIP。
- `full report` 主要来自 GroundingDINO 或 VLM。

并且：

```text
CLIP 不再直接创建正式 OBJECT 节点
```

CLIP 当前只负责产生：

- `view_object_labels`
- `scene_vocabulary`
- `HypothesisPool` 中的短期 hypothesis（`_feed_clip_hypotheses()`）
- `clip_goal_hypotheses` debug cache
- `GoalProposal(source="clip")`
- `frontier_value`

HypothesisPool 已接入主循环，闭环数据流：

```text
每步 CLIP → _feed_clip_hypotheses()
  → hypothesis  score >= 0.22 → HypothesisPool
    → seen_count >= 2, confidence >= 0.45 → "needs_verify"
      → VLM trigger: "hypothesis_verify"
        → VLM 返回 objects
          → promote_by_goal_and_label() → ObjectMemoryPool
          |   → promote 的 hypothesis → OBJECT node in DynamicTopoMap
          → reject_active_by_goal() → 未确认的 hypothesis 被拒绝
```

这意味着：

```text
CLIP = 怀疑有 → HypothesisPool（短期弱证据）
VLM / heavy = 确认有 → ObjectMemoryPool（长期实体记忆）
```

### 3.2 GoalProposal 中间层

当前 planner 已接入轻量 `GoalProposal` 层：

```text
GoalNode
-> generate_goal_proposals()
-> score_goal_proposals()
-> select_best_proposal()
-> planner.navigate_to(proposal.target_position)
```

这层的作用是把三件事分开：

```text
GoalNode = 任务先验
TopoMap node = 环境记忆
GoalProposal = 运行时绑定假设
```

因此多个环境实例不会回流到 GoalGraph，而是表现为多个 proposal 候选。

`source="clip"` 的 proposal 有强规划限制：

- `requires_verification=True`
- `can_stop=False`
- score 有上限，不能和 confirmed object 同级竞争

## 4. 记忆：DynamicTopoMap

ConfTopo 的长期记忆不是纯文本历史，也不是稠密语义地图，而是动态拓扑图：

```text
DynamicTopoMap = nodes + edges + confidence + temporal update
```

主要节点：


| 节点                 | 含义          | 所属层 |
| ------------------ | ----------- | --- |
| WAYPOINT_VISITED   | agent 经过的位置 | Navigation |
| WAYPOINT_FRONTIER  | 可能值得探索的方向   | Navigation |
| WAYPOINT_CANDIDATE | 候选导航点       | Navigation |
| WAYPOINT_APPROACH  | 接近目标的最佳站位   | Navigation |
| OBJECT             | 物体语义节点      | ObjectMemory → Structure/Graph |
| LANDMARK           | 门、走廊、结构性锚点等 | Structure |
| OBJECT_SUMMARY     | 物体摘要（某房间有什么物体） | Structure |
| GOAL_REGION        | 目标优先搜索区域     | Structure |
| ROOM               | 房间或区域摘要     | Structure |


主要边：


| 边              | 含义                         | 所属层 |
| -------------- | -------------------------- | --- |
| NAVIGABLE      | 两个 waypoint 可通行（带 action_hint） | Navigation |
| OBSERVED_AT    | 某个位置观测到某个物体                | Cross |
| BELONGS_TO     | 物体 / waypoint 属于某个 room     | Cross |
| IN_ROOM        | object 位于房间内（显式语义关系）        | Structure |
| NEAR           | object 靠近另一个 object/landmark | Structure |
| ANCHORED_TO    | object 锚定于 waypoint          | Cross |
| SERVES_OBJECT  | ApproachPoint 服务于 OBJECT    | Cross |
| ADJACENT_TO    | room 相邻（门户连接）               | Structure |
| VISIBLE_FROM   | 某个 landmark 可从某位置看到         | Cross |


记忆更新的目标不是“记录所有东西”，而是形成可规划的结构：

```text
哪里走过
哪里没探索
哪里看到过目标相关物体
哪些区域可能属于厨房 / 卧室 / 走廊
哪些门或通道可以作为结构锚点
```

当前 object memory 的写入边界是：

```text
VLM / heavy confirmed observation
-> ObjectObservation
-> upsert_object_observation()
  → ObjectMemoryPool.upsert() → OBJECT node + evidence_refs
  → DynamicTopoMap._sync_approach_point() → WAYPOINT_APPROACH
  → DynamicTopoMap._sync_object_summary_from_room() → OBJECT_SUMMARY
  → DynamicTopoMap._sync_goal_region() → GOAL_REGION
  → IN_ROOM / NEAR / ANCHORED_TO edges (显式语义关系)
```

所以：

- `OBJECT` 节点表示长期确认记忆。
- `Hypothesis` 表示短期怀疑（CLIP 触发，needs_verify 后转 VLM confirm）。
- `ApproachPoint` 表示接近目标的最佳站位。
- `ObjectSummary` 和 `GoalRegion` 表示高级语义结构。

这让：

```text
短期怀疑 → HypothesisPool
长期确认 → ObjectMemoryPool
接近站位 → ApproachPoint
语义摘要 → ObjectSummary / GoalRegion
```

在实现上真正分离。

### Object 节点存储的数据

每个 OBJECT 节点在 attributes 中存储：

```text
evidence_refs: [{step_id, viewpoint_id, bbox, source}]    ← 图像证据引用
object_attributes: {color: {value, confidence}, shape: {...}} ← VLM 属性
bbox_observations: [{bbox, confidence, viewpoint_id, ...}]  ← bbox 历史
detection_scores: [conf1, conf2, ...]
viewpoints: [wp_id_1, wp_id_2, ...]
anchor_waypoint_id / anchor_waypoint_position
bearing / range_bin / visibility
spatial_relation: ["near table", "on counter"]
target_relevance / room_prior_score
best_approach_position
approach_point_id                                     ← 指向 ApproachPoint 节点
```

### Waypoint 节点存储的数据

每个 WAYPOINT_VISITED 节点在 attributes 中存储：

```text
heading: 1.57                                            ← agent 在该位置的朝向（弧度）
semantic_role: "visited_waypoint"
room_id / room_label / room_distance                     ← 所属房间绑定（assign_waypoint_to_room）
waypoint_role: "entrance" | "corridor_anchor"            ← 走廊入口 / 锚点标记（可选）
```

Waypoint 的核心位置存储在 `SemanticNode.position`（以 agent 起点为原点的局部 3D 坐标）。

Heading 在 `_write_waypoint()` 中写入，`MemoryWriter.update()` 传递当前 heading 值。

### 导航边存储的数据

每个 NAVIGABLE 边存储：

```text
distance_m: 1.2
direction_label: "forward"
heading_delta: 0.15
action_hint: "go forward 1.2m, slight right"         ← 自然语言提示
traversability: 1.0
visited_count: 3
blocked_count: 0                                       ← 碰撞计数
last_updated_step: 45
evidence: ["odometry"]
```

### 语义边存储的数据

IN_ROOM / NEAR / ANCHORED_TO 等边存储：

```text
relation: "sink_in_kitchen" | "chair_near_table"
confidence: 0.85
last_updated_step: 42
evidence: ["perception"]
```

### Frontier 节点存储的数据

每个 WAYPOINT_FRONTIER 节点在 attributes 中存储：

```text
semantic_role: "frontier"
anchor_waypoint_id: "wp_12"                              ← 生成该 frontier 的 waypoint（新增）
direction_delta: 0.0 | 1.57 | -1.57 | 3.14                ← 相对 heading 的方向偏移（新增）
frontier_semantic_value: 3.2                              ← 任务相关价值评分（新增，_best_frontier() 中持久化）
consumed: false | true                                     ← 是否已被消耗
blacklisted_until: -1 | step_n                             ← 暂时禁用
```

Frontier 的核心位置也存储在 `SemanticNode.position`。

`_write_frontiers()` 基于当前 heading 在四个方向（forward/left/right/back）生成 frontier，并记录方向偏移。`_best_frontier()` 每步评分后将 `sem_score` 持久化到节点上，避免重复计算。


## 5. 置信度与记忆维护

ConfTopo 中的语义节点不是一次写入后永久可信，而是带有动态置信度。

置信度会影响节点命运：

```text
高置信 / 进入长期记忆（OBJECT node）
中置信 / 保留为候选（可触发 VLM 验证）
低置信 / 衰减 / 删除（prune_low_confidence）
远处但有用 / 折叠保留 anchor（adaptive_granularity）
任务相关 / 保留更久（target_relevance > 0 豁免）
任务无关 / 更快衰减
```

### 5.1 ConfidenceFactors 因子

每个 OBJECT 节点的置信度由多因子综合决定（confidence.py）：

```text
detection_score:      0.25   VLM/CLIP 检测分
multi_view_count:     0.20   多视角确认数
task_relevance:       0.15   当前目标相关性
room_prior_score:     0.10   房间先验匹配
attribute_confidence: 0.10   VLM 属性平均置信度
negative_evidence:   -0.05   弱负面证据（低分观测比例）
strong_negative:     -0.15   强负面证据（VLM 拒绝/验证失败）
time_decay:           0.10   时间衰减
```

### 5.2 置信度公式

```python
raw = (base / max_positive) + frame_bonus + attribute_bonus - penalty
score = max(detection_score, raw) * staleness_decay
```

staleness_decay 包裹整个表达式，保证过期节点整体衰减。

### 5.3 负面证据分层

| 层级 | 触发条件 | 效果 |
|------|---------|------|
| weak negative | 观测分 < 0.2 比例 | max 约 -0.05 |
| strong negative | VLM 分 < 0.3 且 source=vlm | step +0.10，累计 -0.20/unit |
| conflict | 附近不同标签 | max 约 -0.10 |

### 5.4 task-driven 统一分数

compute_task_score() 综合以下因素计算当前目标匹配度：

| 因素 | 权重 | 说明 |
|------|------|------|
| label_match | 0.40 | target_object 精确/部分匹配 |
| attribute_match | 0.15 | 颜色/形状/材质等属性匹配 |
| room_prior_match | 0.20 | 观测房间在 goal.room_prior 中 |
| landmark_relation_match | 0.15 | spatial_relation 提及目标 landmarks |
| history_bonus | 0.10 | 同一目标历史接近成功率 |

Task score 影响：confidence update、planner ranking、stop verification。


ConfTopo 中的语义节点不是一次写入后永久可信，而是带有动态置信度。

置信度会受到以下因素影响：

```text
detection confidence
multi-view consistency
task relevance
room prior
staleness
conflict / redundancy
distance to agent
```

记忆维护包括（含 P11 保护）：

```text
prune_low_confidence 豁免（新增）：
  - OBJECT confidence >= 0.55 → 保留
  - OBJECT multi_view_count >= 2 → 保留
  - 加上已有的 object_anchor / cross_goal_preserved / target_relevance

decay_all_confidences 豁免（新增）：
  - 同上条件

_best_frontier 过滤（新增）:
  - NAVIGABLE edge traversability < 0.2 → 跳过
  - blocked_count > 3 → 跳过
  - blocked_count > 1 → sem_score 惩罚
```

记忆维护包括：

```text
confidence decay
nearby node merge
low-confidence pruning
far object compression
room-level summarization
waypoint-to-room binding
```

这样可以避免拓扑图无限膨胀，也能让远处细碎物体逐渐压缩成更粗粒度的房间/区域摘要。

这也是 ConfTopo 区别于简单 object memory 的关键点：  

> 它维护的是可分层、可衰减、可复用的结构化拓扑记忆。

## 6. 规划：任务驱动图推理

ConfTopo 的 planner 不直接根据当前图像决定动作，而是在 DynamicTopoMap 上选择下一个目标节点。

当前规划采用两阶段思路：

```text
Stage 1: 选择结构锚点
Stage 2: 在结构锚点附近选择导航候选
```

### Stage 1：结构锚点

结构锚点可以是：

```text
ROOM
portal-like LANDMARK
synthetic portal
goal-relevant region
```

例如目标是 `sink`，planner 会倾向于：

```text
kitchen-like room
包含 sink / cabinet / counter 的区域
靠近厨房门口的 frontier
```

### Stage 2：导航候选

候选节点包括：

```text
frontier
candidate waypoint
object node
room summary
visited waypoint fallback
```

评分考虑：

```text
目标物体匹配
room prior
landmark hint
节点置信度
距离代价
访问惩罚
frontier_value
结构锚点 bonus
edge blocked_count penalty
edge traversability skip
```

其中 `frontier_value` 已经显式接入，用于表达：

```text
这个 frontier 是否更像“值得为了当前目标去探索”的方向
```

它主要来自：

- CLIP goal hint
- room prior context
- landmark prior context

Planner 输出：

```text
target_node_id
target_position
is_exploration
mode
scores
debug
```

然后由底层 Navigator 或 PathfinderExecutor 转成实际动作。

### GoalProposal 管道

NavigationPlanner 包含完整的 proposal pipeline：

```text
generate_goal_proposals(candidates, scores, goal_labels)
  → GoalProposal objects with: goal_id, candidate_node_id, target_position, score, can_stop
    → score_goal_proposals(proposals, goal, topo_map)
      → Unified scoring: score + task_score + memory_reuse_bonus
        → select_best_proposal(proposals)
          → max(active proposals by score)
```

GoalProposal 引用已有 node，不新建重复 goal 实例。

### reachability 估算

```python
_compute_reachability(candidates, position):
    reach = [1.0] * len(candidates)             # 默认可达
    cost = [min(d / 20.0, 1.0) for d in dists]   # 路径代价
```

proposal_score 综合因子：

```text
proposal_score =
  goal_match_score    (compute_task_score)
  + confidence        (node.confidence)
  + reachability      (reachable = 1.0, unreachable = 0.0)
  + memory_reuse_bonus(score_goal_proposals 中)
  + frontier_value    (frontier_semantic_value)
  - distance_cost     (归一化路径代价)
```

### Approach / Stop 状态机

NavPhase 状态机管理 8 个导航阶段：

```text
GLOBAL_SEARCH
  → ROUTE_TO_STRUCTURE (room/landmark 目标)
  → ROUTE_TO_OBJECT_ANCHOR (object anchor 目标)
    → LOCAL_VISUAL_APPROACH (接近目标)
      → STOP_VERIFY (验证停)
        → STOP (停)
      → RECOVERY (目标丢失)
```

停条件（3 层验证）：

```text
layer1: confirm_buffer 多帧确认目标可见
layer2: bbox 增长 + approach 距离 + plateau creep
layer3: fresh VLM + stop_candidate + 居中 + visible + task_score > 0.3
final: layer1 && layer2 && layer3 && !retreating
```

保守策略：range_bin in ("very_near", "close") 才可能停，near 不停。

## 7. 多目标记忆复用

GOAT-Bench 的关键是 multi-goal episode。ConfTopo 在 goal 切换时：

```text
保留 DynamicTopoMap（_mark_cross_goal_preserved + reset_keep_memory）
保留已探索结构（ROOM / LANDMARK / WAYPOINT 全部保留）
保留 object / landmark memory（含 strong_negative_evidence 标记）
保留 ApproachPoint / ObjectSummary / GoalRegion（若有保留的 source）
清空当前 goal 的短期规划状态（trigger_state / servo_state / recovery）
重新设置 goal labels / landmarks / room prior
```

也就是：

```text
episode reset -> 清空记忆（topo_map.reset()）
goal switch -> 不清空记忆（reset_keep_memory() 只重置 step_count）
```

### 7.1 跨目标保留的节点

| 节点类型 | 保留条件 |
|---------|---------|
| OBJECT | object_anchor / context_object / target_relevance > 0 / strong_neg <= 0.3 |
| ROOM | 无条件保留 |
| LANDMARK | 无条件保留 |
| WAYPOINT_VISITED | prune 豁免 |
| WAYPOINT_APPROACH | 若 source_object_id 被保留 |
| OBJECT_SUMMARY | 若 room_node_id 被保留 |
| GOAL_REGION | 若 room_node_id 被保留 |

### 7.2 记忆复用机制

`_scan_reuse()` 在新目标开始时检索现有 OBJECT 节点：

```python
same_label → 匹配标签的 object
  ├── blacklisted_until >= step → failed_active（暂时不可用）
  ├── strong_negative_evidence > 0.5 → failed_expired（VLM 拒绝过）
  ├── failed_approach_count > 0 → failed_expired（历史接近失败）
  └── else → unfailed（可直接复用）
```

Planner 在 `_best_object_anchor()` 中利用这些信息调整评分：

```text
unfailed_anchors（重复目标时）      +0.6
failed_approach_count > 0          -0.5
strong_negative_evidence            -0.50 * value
cross_goal_preserved                +0.2
attribute_confidence                +0.05
```

### 7.3 举例

```text
Goal 1: find sofa
探索过程中建立 living room / sofa / table / window 结构
sofa 的 OBJECT 节点置信度提升，成为 object_anchor

Goal 2: find TV
_scan_reuse() 没有找到 TV 的 memory hit
但 living room ROOM 被保留
planner 优先选择通往 living room 方向的 frontier
实现"不需要重新乱搜，优先回相关区域"
```

这正是 ConfTopo 在多目标任务中的核心优势。

## 8. VLM 在 ConfTopo 中的位置

VLM 不是替代整个导航系统，而是替代或增强“怎么看环境”这一层。

推荐定位：

```text
VLM = 结构化场景理解器
DynamicTopoMap = 长期世界模型
Planner = 图推理决策器
Navigator = 动作执行器
```

VLM 负责回答：

```text
当前是什么房间？
看到了哪些物体？
目标是否可能可见？
哪里像门、走廊、通道？
当前区域是否和目标相关？
```

VLM 不应该负责每步直接输出低层动作，因为这样会丢掉 ConfTopo 的长期记忆和图推理优势。

更合理的表达是：

```text
VLM perception grounds semantics into topo memory.
Topo memory supports long-horizon multi-goal planning.
```

## 9. 与 GOAT-Bench 的关系

正式实验应使用 GOAT-Bench 官方框架负责：

```text
dataset
split
episode
goal switch
env
action loop
metric
benchmark output
```

ConfTopo 只作为 agent brain 接入：

```text
GOAT-Bench official runner
    +
ConfTopoAgent wrapper
```

不要重写 benchmark metric，也不要把自定义 trace runner 当作最终评测依据。

自定义脚本适合用于：

```text
debug
trace
visualization
acceptance smoke test
ablation prototype
```

正式结果应回到官方 runner / env / metric。

## 10. 方法创新点

ConfTopo 可以从四个角度表述创新：

### 10.1 结构化长期拓扑记忆

不是保存单帧 caption 或 object list，而是把 waypoint、object、room、landmark 和 frontier 写成动态拓扑图。

### 10.2 任务驱动的语义图规划

规划不是纯几何 shortest path，也不是纯 VLM action prediction，而是根据当前 goal 在拓扑图上选择语义相关的下一步目标。

### 10.3 多目标记忆复用

在同一 episode 内，goal 切换不清空地图，使后续目标可以复用已探索结构和语义节点。

### 10.4 触发式 VLM 感知

VLM 只在低置信度、接近目标、卡住、reground 等关键时刻调用，平衡语义能力和在线效率。

## 11. 推荐论文表达

可以把方法写成：

```text
We propose ConfTopo, a confidence-aware dynamic topological memory framework
for multi-goal object navigation. ConfTopo converts egocentric visual
observations into structured perception reports, incrementally builds a
semantic topological map, and performs task-driven graph planning over the
memory to support long-horizon goal switching and memory reuse.
```

中文表述：

```text
我们提出 ConfTopo，一种面向多目标对象导航的置信度感知动态拓扑记忆框架。
该方法将单视角视觉观测转化为结构化感知报告，持续构建包含 waypoint、
object、room、landmark 和 frontier 的语义拓扑图，并在该图上执行任务驱动
的结构化规划，从而支持长时程探索、多目标切换和记忆复用。
```

## 12. 推荐实验与消融

主实验：

```text
GOAT-Bench official Modular baseline
ConfTopo with CLIP + GroundingDINO
ConfTopo with VLM perception
```

消融实验：


| 实验                              | 目的                |
| ------------------------------- | ----------------- |
| w/o structured memory           | 验证长期拓扑记忆作用        |
| w/o confidence maintenance      | 验证置信度衰减 / 合并 / 剪枝 |
| w/o two-stage planner           | 验证结构锚点规划作用        |
| w/o VLM                         | 验证 VLM 语义理解增益     |
| VLM every step vs triggered VLM | 验证触发式调用的效率优势      |
| reset memory per goal           | 验证多目标记忆复用         |


核心指标：

```text
SR
SPL
softSPL
steps per goal
memory reuse count
VLM calls per episode
object/room node precision
```

## 13. 一句话总结

ConfTopo 的方法本质是：

```text
用 VLM/CLIP/GDINO 看环境，
用 DynamicTopoMap 记环境，
用任务驱动图规划想下一步，
用 Navigator 执行动作。
```

它不是一个单纯的 VLM agent，也不是一个普通拓扑地图，而是一个：

```text
感知可替换
记忆可复用
规划可解释
评测可接入官方 benchmark
```

的结构化多目标导航框架。
