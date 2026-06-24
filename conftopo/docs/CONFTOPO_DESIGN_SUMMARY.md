# ConfTopo 项目总体设计总结

## 1. 项目定位

ConfTopo 可以定义为：

> **Task-driven, Confidence-aware Dynamic Semantic TopoMap for Human-like Zero-shot Navigation**

中文表述为：

> **面向拟人导航的任务驱动、置信度感知、动态语义拓扑记忆地图。**

它的核心不是单纯构建一个更复杂的 TopoMap，而是把导航过程组织成一种更接近人类认知的闭环：

```text
理解任务
-> 快速观察环境
-> 形成怀疑线索
-> 多次确认后固化成记忆
-> 根据长期记忆复用路线和语义经验
-> 接近目标后再次确认是否停止
```

对应到系统模块：

| 认知过程 | ConfTopo 模块 |
| --- | --- |
| 理解任务 | LLM GoalGraph |
| 快速扫视 / 粗筛 | CLIP / LightPerceiver |
| 仔细确认 | VLM / HeavyPerceiver |
| 短期怀疑 | Hypothesis Pool |
| 长期语义实体记忆 | Object Memory Pool |
| 语义认知地图 | Structure Layer |
| 可行走拓扑图 | Navigation Layer |
| 记忆可信度 | Confidence |
| 经验复用 | Long-term Reuse |


因此，ConfTopo 最终可以表述为：

> 通过任务驱动的置信度记忆固化与长期复用，实现一种更接近人类认知过程的 zero-shot 多目标导航系统。

## 2. 总体系统流程

完整流程：

```text
Goal
-> LLM
-> GoalGraph
-> CLIP + VLM Perception
-> Hypothesis Pool (CLIP hint → needs_verify → VLM trigger)
-> Object Memory Pool (VLM confirmed → promote hypothesis)
-> Dynamic Semantic TopoMap
   |-> Structure Layer (Room / Landmark / ObjectSummary / GoalRegion)
   |   |-- IN_ROOM edges (object ↔ room)
   |   |-- NEAR edges (object ↔ object/landmark)
   |   `-- ADJACENT_TO edges (room ↔ room)
   `-> Navigation Layer (Waypoint / Frontier / ApproachPoint)
       |-- NAVIGABLE edges (action_hint / distance / traversability)
       `-- ANCHORED_TO edges (approach_point ↔ object)
-> Planner
-> Route to ApproachPoint
-> Stop Verification
```

简化为：

```text
任务理解 -> 感知确认 -> 记忆建图 -> 任务驱动规划 -> 接近目标 -> 停止验证
```

当前代码中的主闭环对应：

```text
observe -> update_memory -> plan -> act
```

主入口：

```text
conftopo/agents/goat_agent_new.py
```

核心记忆结构：

```text
conftopo/core/dynamic_topo_map.py
```

## 3. GoalGraph 设计与修改方向

### 3.1 GoalGraph 的定位

GoalGraph 是由 LLM 从原始任务目标中解析出的结构化任务先验：

```text
GoalGraph = Task Prior
```

例如：

```text
find the red chair near the dining table
```

应解析为：

```text
target_object: chair
attributes: red
room_prior: dining room
landmarks: dining table
relations: near(table)
```

它的作用不是直接导航，也不是生成环境中的多个物理实例，而是给感知、记忆、规划和停止验证提供语义约束。

### 3.2 GoalGraph 应该驱动什么

GoalGraph 应持续影响整个导航闭环：

- 感知词表
- CLIP 检索目标
- VLM prompt
- Object / Room / Landmark 打分
- Frontier 价值评估
- Planner 候选选择
- Stop verification

也就是说，不能只是“解析一次目标”，而应该让当前任务持续影响：

```text
perception -> memory -> planning -> approach -> stop
```

### 3.3 GoalGraph 不生成物理实例

GoalGraph 只表达目标语义，不表达环境中的多个物理实例。

正确关系是：

```text
一个 GoalNode
-> 多个 ObjectNode 候选
-> 多个 GoalProposal
-> Planner 选择最可信的一个
```

如果导航中出现多个相同实例，问题通常不在 GoalGraph，而是在后面的：

- Object merge
- GoalProposal 去重
- Object-Goal 绑定关系
- Candidate ranking

因此不要把多个 chair / sink / table 实例塞回 GoalGraph。

### 3.4 当前实现是否符合这个设定

当前 `goat_agent_new.py` 整体方向是符合的：

- `GoalNode` 表达 `target_object / attributes / room_prior / landmarks / relations` 等任务语义字段。
- `DynamicTopoMap` 中的 `OBJECT` 节点负责表达环境里实际看到的物理候选。
- `goat_agent_new.py` 使用当前 `GoalNode` 设置感知词表、VLM 上下文和 planner 打分。
- 运行时类别先验保存在 `GoalManager.runtime_room_prior / runtime_landmarks`，不会直接改写原始 `GoalNode.room_prior / landmarks`。

但还需要补强四点。

### 3.5 需要修改的地方

**第一，LLM 负责输出干净的 target_object。**

推荐：

```text
target_object = "chair"
attributes = ["red"]
```

不推荐：

```text
target_object = "red chair"
```

原因是 `target_object` 会进入 CLIP / VLM / detector 词表和 object matching。如果把属性混进类别名，会影响 object merge、goal label match 和 stop verification。

注意：这一步应该由 LLM GoalGraph parser 完成，而不是由 `instruction_graph.py` 用规则兜底拆词。代码层只做 schema normalization / validation，不能擅自把 `white chair` 改成 `chair`。

**第二，不要在运行时直接改写原始 GoalGraph。**

当前 agent 会在 goal 缺少 `room_prior / landmarks` 时回填类别先验。这个逻辑有用，但更清晰的边界是：

```text
LLM GoalGraph = 原始任务先验
Runtime GoalContext = agent 推断出的补充先验
```

例如：

```text
GoalGraph:
  target_object: sink

Runtime GoalContext:
  inferred_room_prior: [kitchen, bathroom]
  inferred_landmarks: [counter, cabinet]
```

这样可以避免把“LLM 解析结果”和“agent 常识补全”混在一起。

**第三，让 relations 真正参与打分。**

当前 `relations` 字段已经存在，但还没有充分用于 planner scoring。

例如：

```text
target_object: chair
relations: near(table)
```

应影响候选选择：

```text
chair ObjectNode near table ObjectNode / LandmarkNode
  -> relation_match_score 高

chair ObjectNode 没有 table 证据
  -> relation_match_score 低
```

这一步应该放在 GoalGraph 查询 TopoMap / GoalProposal scoring 中，而不是让 GoalGraph 自己生成实例。

**第四，增加 GoalProposal 层。**

GoalProposal 是 GoalGraph 和 TopoMap 之间的候选绑定结果：

```text
GoalProposal(
    goal_id,
    object_node_id / room_node_id / frontier_id,
    score,
    reason,
    evidence_refs
)
```

注意：

```text
Proposal 引用已有 node
不要新建重复 goal instance
```

它可以统一承载：

- goal_match_score
- attribute_match_score
- relation_match_score
- confidence
- reachability
- distance
- memory_reuse_value

### 3.6 当前代码边界

当前主实现已经按下面的边界收敛：

```text
GoalNode
= 任务先验
= 我要找什么

ObjectNode / Room / Frontier / Waypoint
= 环境记忆
= 我看到了什么 / 我去过哪里 / 哪里还没探索

GoalProposal
= 运行时假设
= 当前这个环境候选是不是这个目标
```

对应到代码：

- `GoalNode` 只保留任务语义字段，不保存环境实例。
- `GoalProposal` 作为 planner 的中间候选单位，绑定当前 goal 和已有 map node。
- `GoalProposal.source` 用于区分候选来源，例如 `object_memory / room_prior / frontier / clip`。

### 3.7 Runtime GoalContext

当前代码明确区分了：

```text
LLM GoalGraph
Runtime GoalContext
```

含义是：

- `GoalGraph` 保存 LLM 解析出的原始任务先验。
- `Runtime GoalContext` 保存 agent 在运行时补充的类别常识先验。

例如：

```text
GoalGraph:
  target_object: sink

Runtime GoalContext:
  inferred_room_prior: [kitchen, bathroom]
  inferred_landmarks: [counter, cabinet]
```

因此：

- `goal.room_prior / goal.landmarks` 不在运行时被直接改写。
- 规划、VLM prompt、heavy labels 读取的是 effective prior，而不是污染原始 GoalGraph。

## 4. 感知设计：CLIP + VLM

### 4.1 设计原则

感知分工明确为：

```text
CLIP = lightweight semantic proposal
VLM / HeavyPerceiver = semantic verification
```

也就是：

```text
CLIP 负责“怀疑有”
VLM 负责“确认有”
```

这意味着：

- CLIP 适合高频运行，每步提供轻量语义线索。
- VLM 低频触发，用于可靠确认、属性提取和 stop verification。

### 4.2 CLIP 当前职责

当前代码中，CLIP / LightPerceiver 每步运行，主要负责：

- `room_scores / room_label`
- `goal_scores`
- `landmark_scores`
- waypoint 视角上下文更新
- `clip_goal_hypotheses`
- `frontier_value` 的显式估计

CLIP 的输出不会直接创建正式 `ObjectNode`。

当前约束是：

```text
CLIP goal score
-> clip_goal_hypotheses
-> GoalProposal(source="clip")
-> frontier_value / planner boost

不会：
-> 直接写 Object Memory
```

这条边界非常关键，因为它把：

```text
怀疑线索
```

和：

```text
长期确认记忆
```

区分开了。

### 4.3 VLM / Heavy 当前职责

VLM / HeavyPerceiver 低频触发，用于：

- object verification
- landmark verification
- room verification
- bbox
- range_bin
- visibility
- confidence
- attributes
- stop_candidate

也就是说，正式对象写入遵循：

```text
VLM / heavy confirmed observation
-> ObjectObservation
-> upsert_object_observation()
-> DynamicTopoMap OBJECT node
```

当前 ObjectObservation 已保留：

- `label`
- `bbox`
- `confidence`
- `range_bin`
- `bearing`
- `visibility`
- `spatial_relation`
- `room_context`
- `attributes`

其中 `attributes` 用于保存例如：

```text
color / material / other VLM-extracted properties
```

### 4.4 Hypothesis Pool 的当前落点

`HypothesisPool` 模块已接入主循环（`goat_agent_new.py`），形成完整闭环：

```text
PerceptionManager.run()
  |
  run_light() → CLIP goal_scores
    |
    _feed_clip_hypotheses() → HypothesisPool.add_or_update()
      |
      得分 >= 0.22 时创建/更新 hypothesis（kind=object, source=clip）
      多帧稳定 → status = "needs_verify"
    |
    _should_run_vlm()
      |-- 常规触发器（间隔 / 低置信 / frontier / summary）
      |-- 新增: hypothesis_verify 触发器
      |      当 pool 中存在 needs_verify 的 hypothesis 时，
      |      以 heavy_interval // 2 的频率触发 VLM
    |
    VLM 返回 result
      |-- 匹配 VLM objects → promote_by_goal_and_label()
      |-- 未匹配的 needs_verify → GoatAgent._update_hypotheses_from_memory() reject
      |-- promoted hypothesis → ObjectMemoryPool.upsert() → OBJECT node
      |-- rejected hypothesis → 冷却期后过期
```

Hypothesis 数据结构：

```text
Hypothesis(id, goal_id, kind, label, source, anchor_node_id, position,
           score, confidence, weak_bbox, weak_relations,
           first_seen_step, last_seen_step, seen_count, ttl, status)

status: "active" → "needs_verify" → "promoted" | "rejected" → "expired"
```

关键集成点：

| 集成点 | 位置 | 作用 |
|-------|------|------|
| CLIP → Hypothesis | `PerceptionManager.run()` → `_feed_clip_hypotheses()` | 每步将 CLIP 得分写入 hypothesis |
| Hypothesis → VLM trigger | `_should_run_vlm()` → `hypothesis_verify` | needs_verify 触发 VLM 验证 |
| VLM → promote | `GoatAgent._update_hypotheses_from_memory()` | VLM 写入 memory 后提升匹配 hypothesis |
| VLM → reject | `GoatAgent._update_hypotheses_from_memory()` | 未确认的 needs_verify 被拒绝 |
| planner hint | `NavigationPlanner._best_frontier()` | hypothesis 位置参与 frontier 打分 |
| decay | `hypothesis_pool.decay(current_step)` | 每步衰减 TTL / confidence |
| debug | `DebugTracer` + `memory_stats` | 输出 hypothesis pool 状态 |

边界：

- Hypothesis 不直接创建 OBJECT 节点（只能被 promote 后指向已有 object）
- Hypothesis 不能触发 stop
- Hypothesis 不参与长期 memory reuse

当前 `goat_agent_new.py` 的实现边界：

- CLIP 只写入 `HypothesisPool`，并生成 `GoalProposal(source="clip")` 弱提示。
- CLIP proposal 被 `CLIP_PROPOSAL_SCORE_CAP` 限分，并强制 `can_stop=False / requires_verification=True`。
- VLM 低置信 object 写入 `Hypothesis(source="vlm_weak")`，保存 `weak_bbox / weak_relations / attributes`，不直接进入正式地图。
- VLM confirmed object 通过 `MemoryWriter._write_objects()` 写入 `DynamicTopoMap.upsert_object_observation()`，并保存 attributes。
- memory 写入完成后，`GoatAgent._update_hypotheses_from_memory()` 显式执行 hypothesis promote / reject。

### 4.5 Frontier Value 的当前定义

当前 frontier 不再只是“没探索过的点”，而是显式带有任务相关价值：

```text
frontier_value =
  clip goal hint
  + room prior context
  + landmark prior context
  + hypothesis positions (from HypothesisPool)
  + frontier_semantic_value (persisted per node)
```

因此 planner 不只是看空间可达性，还会看：

```text
这个 frontier 是否更像“值得为了当前目标去探索”的方向
```

### 4.5.1 Frontier 数据存储

每个 frontier 节点在 `_write_frontiers()` 中写入以下 attributes：

```text
semantic_role: "frontier"
anchor_waypoint_id: cur_vp                           ← 生成该 frontier 的 waypoint
direction_delta: 0.0 | 1.57 | -1.57 | 3.14           ← 相对 heading 的方向偏移（rad）
frontier_semantic_value: float                       ← 任务相关价值（NavigationPlanner 中持久化）
consumed: false
blacklisted_until: -1
```

`NavigationPlanner._best_frontier()` 每步评分后将 `frontier_semantic_value` 写回节点，避免下次查询时重复计算全部 frontier 的语义价值。

### 4.6 当前实现总结

因此当前 ConfTopo 的感知-记忆边界可以总结为：

```text
CLIP (LightPerceiver, every step)
  -> goal_scores / room_scores / landmark_scores
  -> HypothesisPool (add_or_update)
  -> frontier_value / planner hint

HypothesisPool (TTL-managed weak evidence)
  -> feed_clip_hypotheses → seen_count >= 2 → needs_verify
  -> hypothesis_verify trigger → VLM invoked
  -> promote_by_goal_and_label → ObjectMemoryPool
  -> _update_hypotheses_from_memory reject → cooldown / expired hypothesis

VLM / HeavyPerceiver (triggered）
  -> ObjectObservation (label, bbox, attributes, confidence)
  -> stop_candidate / goal_visible / target_direction
  -> PerceptionReport (full)

ObjectMemoryPool (entity-level object management)
  -> upsert(): create / merge / match objects
  -> compute_best_approach(): approach point computation
  -> promote_anchor_if_better(): anchor waypoint promotion
  -> metadata: evidence_refs, bbox_observations, viewpoints

DynamicTopoMap (graph-level topological memory)
  -> Structure Layer: Room, Landmark, ObjectSummary, GoalRegion
  -> Navigation Layer: Waypoint, Frontier, ApproachPoint
  -> IN_ROOM / NEAR / ANCHORED_TO / SERVES_OBJECT edges
  -> edge metadata: action_hint, relation, confidence, traversability
```
- stop_ready

Planner 最终选择的是 Proposal，而不是直接让 GoalGraph 决定动作。

## 5. ObjectMemoryPool：对象实体记忆

ObjectMemoryPool 从 DynamicTopoMap 中抽出，独立管理对象实体生命周期。

### 5.1 职责边界

```text
ObjectMemoryPool           DynamicTopoMap
───────────────            ──────────────
upsert / merge / match     图边（OBSERVED_AT, IN_ROOM, SERVES_OBJECT）
bbox_observations          节点创建（通过回调 add_node）
viewpoints / anchor          navigation layer 视图
confidence computation        prune / decay / granularity
approach position            sync_approach_point → WAYPOINT_APPROACH
landmark promotion           sync_object_summary → OBJECT_SUMMARY
evidence_refs / attributes   sync_goal_region → GOAL_REGION
```

### 5.2 Object 属性与证据

每个 OBJECT 节点记录：

```text
attributes:
  evidence_refs: [{step_id, viewpoint_id, bbox, source}]
  object_attributes: {color: {value, confidence}, shape: {...}, ...}
  bbox_observations: [{bbox, confidence, viewpoint_id, ...}]
  detection_scores: [conf1, conf2, ...]
  viewpoints: [wp_id_1, wp_id_2, ...]
  anchor_waypoint_id / anchor_waypoint_position
  bearing / range_bin / visibility
  spatial_relation: ["near table", "on counter"]
  target_relevance / room_prior_score
  best_approach_position
```

### 5.3 VLM 属性提取

VLM prompt 的 JSON schema 包含每物体的 `attributes` 字段：

```json
{
  "label": "chair",
  "bbox": [0.1, 0.2, 0.5, 0.8],
  "attributes": {
    "color": {"value": "red", "confidence": 0.85},
    "shape": {"value": "dining chair", "confidence": 0.7},
    "material": {"value": "wood", "confidence": 0.55},
    "size": {"value": "medium", "confidence": 0.8},
    "state": {"value": "intact", "confidence": 0.6},
    "description": {"value": "a red wooden dining chair", "confidence": 0.5}
  },
  "confidence": 0.82
}
```

每个属性带独立置信度，避免 VLM 单次描述被当成绝对事实。

## 6. ApproachPoint：接近目标节点

`WAYPOINT_APPROACH` 是 Navigation Layer 的独立节点，由 `_sync_approach_point()` 在 object upsert 后自动创建。

### 6.1 生命周期

```text
ObjectMemoryPool.upsert() 计算 best_approach_position
  → DynamicTopoMap._sync_approach_point()
    → 首次：创建 WAYPOINT_APPROACH 节点 + SERVES_OBJECT(→OBJECT) + NAVIGABLE(→waypoint)
    → 后续：位置变化 > 0.15m 时更新节点位置
    → 通过 node.attributes["approach_point_id"] 双向绑定
```

### 6.2 作用

```text
之前：NavigationPlanner 读取 object.best_approach_position attribute
现在：Navigation Layer 直接路由到 ApproachPoint 节点

ApproachPoint 属性：
  source_object_id / object_label
  position (best approach position)
  NAVIGABLE 边连接最近的 waypoint（参与路径规划）
  SERVES_OBJECT 边指向目标 OBJECT（跨层语义连接）
```

## 7. Structure Layer 高级节点

### 7.1 ObjectSummaryNode

`OBJECT_SUMMARY` 节点保存房间级别的物体摘要，由 `_sync_object_summary_from_room()` 自动创建。

```text
触发条件：OBJECT 节点 upsert 后，其 room_context 匹配的 ROOM 节点存在
创建：OBJECT_SUMMARY --BELONGS_TO→ ROOM
属性：
  contains_labels: {chair: 2, table: 1, sink: 1}
  room_node_id
更新：room 内新物体加入后自动追加
用途：StructurePlanner 候选、语义地图査询
```

### 7.2 GoalRegionNode

`GOAL_REGION` 节点标记当前目标应优先搜索的区域。

```text
触发条件：写入 object_anchor 时
创建：GOAL_REGION --BELONGS_TO→ 最近的 ROOM
属性：
  goal_key / target_label / room_label / room_node_id
  confidence (≥ 0.6)
更新：相同 goal_key 的已有节点 → 提升 confidence
用途：StructurePlanner 选择结构锚点时的优先候选
```

## 8. Edge 数据结构

### 8.1 导航边（NAVIGABLE）

```text
distance_m          : float    实际行进距离
direction_label     : "forward" | "left" | "right" | ...
heading_delta       : float    相对航向差
action_hint         : "go forward 1.2m, slight right"
traversability      : 0.1-1.0  可通行性（碰撞后衰减）
visited_count       : int      经过次数
blocked_count       : int      碰撞阻塞次数（新增）
last_updated_step   : int      最后更新时间
evidence            : ["odometry", "perception"]
```

### 8.2 语义边（IN_ROOM / NEAR / ANCHORED_TO / SERVES_OBJECT）

```text
relation            : "sink_in_kitchen" | "chair_near_table"
confidence          : 0.0-1.0  关系可信度
last_updated_step   : int      最后更新时间
evidence            : ["perception"]
```

### 8.3 BLOCKED 处理

`increment_edge_blocked()` 在碰撞事件发生时递增 `blocked_count`，使 planner 可以避开频繁阻塞的路径。

### 3.6 推荐修改顺序

优先级从高到低：

1. 规范 LLM prompt / parser 输出：`target_object` 只保留干净类别名，属性放入 `attributes`。
2. 增加 GoalGraph schema normalization / validation：只清洗类型和空格，不用规则重写语义字段。
3. 把运行时补全的 room / landmark prior 从 GoalGraph 中拆出来，放到 Runtime GoalContext。
4. 在 scoring 中加入 relation_match_score，让 `near(table)` 真正影响候选排序。
5. 增加 GoalProposal 数据结构，统一 candidate ranking 和去重。
6. 让 stop verification 使用 GoalGraph 的 target / attributes / relations 做最终确认。

## 9. Confidence-aware 设计

### 9.1 ConfidenceFactors

`ConfidenceFactors` (`confidence.py`) 包含以下因子：

```python
detection_score: float       # VLM/CLIP 检测置信度
multi_view_count: int        # 多个不同 viewpoint 确认
multi_frame_consistency: int # 同一视角连续确认次数（仅 fresh 观测）
task_relevance: float        # 与当前目标的相关性
room_prior_score: float      # 物体是否在目标房间先验中
attribute_confidence: float  # VLM 属性（color/shape/material）平均置信度

# 负面证据（新增）
negative_evidence: float          # weak negative: 低置信观测的比例
strong_negative_evidence: float   # strong negative: VLM 明确否定 / 验证失败

# 时空衰减
time_decay: float
staleness_steps: int

# 惩罚项
redundancy_penalty: float   # 重复弱观测
conflict_penalty: float     # 附近标签冲突
```

`ObjectMemoryPool._merge_observation()` 自动追踪这些因子。

### 9.2 置信度公式

```text
C_semantic = max(detection_score, weighted_base) * staleness_decay

其中:
  weighted_base = w_det*detection + w_mv*multi_view + w_tr*task_relevance + w_rp*room_prior
  + frame_bonus + attribute_bonus
  - weak_neg_penalty - strong_neg_penalty - conflict_penalty

weights:
  detection: 0.25 | multi_view: 0.20 | task_relevance: 0.15
  room_prior: 0.10 | attribute: 0.10
  negative_evidence: -0.05 | strong_negative_evidence: -0.15
  time_decay: 0.10
```

注意：staleness_decay 包裹整个表达式，惩罚项先于衰减应用。

### 9.3 负面证据分层

| 类型 | 触发条件 | 惩罚强度 |
|------|---------|---------|
| weak negative | detection_score < 0.2 的比例 | * -0.05 |
| strong negative | VLM 明确看见但置信度 < 0.3 | step +0.10，累计 * -0.20 |
| conflict | 附近不同标签的 object | * -0.10 |

Strong negative 随时间衰减（VLM 再次确认 >= 0.3 时 -0.05/step）。

### 9.4 Hypothesis / Confirmed 晋升

P0 已实现（HypothesisPool / VLM trigger / promote/reject），晋升标准：

- VLM confirm（promote_by_goal_and_label()）
- 多帧一致（multi_frame_consistency 累计）
- 任务高相关（task_relevance 影响 confidence）
- 多视角一致（multi_view_count 累计）
- Anchor 稳定（same viewpoint_id 重复观测）

删除条件：

- 长时间没看到（staleness_steps 衰减）
- VLM 拒绝（_update_hypotheses_from_memory() / strong_negative_evidence）
- 多次冲突（conflict_penalty）
- 任务无关且低置信（prune_low_confidence()）

### 9.5 ObjectMemoryPool / DynamicTopoMap 边界

当前收口后的边界是：

```text
ObjectMemoryPool
  = object entity memory
  = merge / bbox history / attributes / confidence / anchor / memory_state

DynamicTopoMap
  = graph representation
  = nodes / edges / room relation / navigation relation / structure layer
```

因此：

```text
ObjectMemoryPool 中的 object memory
  -> 同步为 DynamicTopoMap 的 OBJECT node reference
  -> 通过 IN_ROOM / NEAR / ANCHORED_TO / OBSERVED_AT 等边进入图结构
```

`OBJECT` node 上会保存：

```text
memory_owner = "ObjectMemoryPool"
graph_role = "object_node_ref"
memory_state = candidate / confirmed / preserved / rejected / expired
```

confidence update 会触发 memory state transition：

```text
new observation
  -> update confidence factors
  -> compute_semantic_confidence()
  -> update_memory_state()
  -> candidate / confirmed / preserved / rejected / expired
```

## 10. Task-driven 设计

### 10.1 贯穿范围

当前目标持续影响：

| 环节 | 接入方式 | 状态 |
|------|---------|------|
| CLIP 词表 | LightPerceiver.set_goal_labels() | / |
| VLM prompt | GoalManager.goal_context() | / |
| Hypothesis 保留 | HypothesisPool 的 goal_id 筛选 | / |
| Object confidence | task_relevance + attribute_confidence + negative_evidence | / |
| Room relevance | room_prior_score (binary, 0/1) | ( 可改进) |
| Landmark relevance | _context_object_positions() | / |
| Frontier value | `_collect_frontier_proposals()` 含 hypothesis + room_prior + landmark | / |
| Candidate ranking | `score_goal_proposals()` 统一评分 | / |
| Stop verification | task_score > 0.3 条件 + attribute/relation match | / |
| Memory pruning | target_relevance > 0 豁免 | / |

### 10.2 compute_task_score 统一函数

```python
def compute_task_score(*, target_object, attributes, observed_label,
                        observed_attributes, observed_room, observed_relations,
                        room_prior, landmarks, relations, map_relation_labels,
                        history_success_count, history_fail_count) -> float:
```

因素权重：

| 因素 | 权重 | 说明 |
|------|------|------|
| label_match | 0.40 | 标签精确/部分匹配 |
| attribute_match | 0.15 | VLM 属性与目标属性匹配（color/shape/material/size/state） |
| room_prior_match | 0.20 | 观测房间在 goal.room_prior 中 |
| landmark_relation_match | 0.15 | spatial_relation / graph relation 提到目标 landmarks 或 GoalGraph relations |
| history_bonus | 0.10 | 同一目标历史接近成功率（仅基于 agent 自认成功） |

当前实现拆成两个小 scorer：

```text
AttributeMatcher:
  GoalGraph.attributes vs ObjectMemoryPool.object_attributes

RelationScorer:
  GoalGraph.relations / landmarks / room_prior
  vs spatial_relation / NEAR / IN_ROOM / view_object_labels
```

### 10.3 NavigationPlanner 评分增强

```python
_best_object_anchor() 评分:
  score = confidence + target_relevance
  + attribute_confidence * 0.05      # 小加分：属性清晰度
  - negative_evidence * 0.20         # 弱负面惩罚
  - strong_negative_evidence * 0.50  # 强负面惩罚（约等于否决）
```

### 10.4 StopVerifier 增强

```python
task_score_stop = compute_task_score(
    target_object=goal.target_object,
    observed_label=best_obs.label,
    observed_attributes=best_obs.attributes,
)
fresh_stop_ok = (
    packet.fresh_vlm
    and goal_visible
    and task_score_stop > 0.3   # task-driven stop 条件
    ...
)
```

## 11. 长期记忆保存与复用

### 11.1 Goal 切换时的记忆保留

Goal 切换调用 `set_new_goal()` → `_mark_cross_goal_preserved()`，不清空 `DynamicTopoMap`。

| 节点类型 | 保留策略 | 状态 |
|---------|---------|------|
| OBJECT (object_anchor / context_object) | `cross_goal_preserved = True` | ✅ |
| OBJECT (target_relevance > 0) | `cross_goal_preserved = True` | ✅ |
| OBJECT (strong_negative_evidence <= 0.3) | `cross_goal_preserved = True` | ✅ |
| ROOM | 全部保留 | ✅ |
| LANDMARK | 全部保留 | ✅ |
| WAYPOINT_VISITED | prune 豁免 | ✅ |
| WAYPOINT_APPROACH | 若 source_object_id 被保留 | ✅ |
| OBJECT_SUMMARY | 若 room_node_id 被保留 | ✅ |
| GOAL_REGION | 若 room_node_id 被保留 | ✅ |
| WAYPOINT_FRONTIER | prune 会删除，但 waypoint 保留后可重新生成 | ⚠️ |
| Blocked edges (blocked_count) | edge data 保留但 planner 未显式查询 | ❌ |

### 11.2 记忆复用机制

`GoalManager.set_new_goal()` 调用 `_scan_reuse()`，在现有 OBJECT 节点中检索：

```python
reuse_debug = {
    "unfailed_anchors": [...],       # 匹配 label + 无 failed_approach
    "failed_active_anchors": [...],  # blacklisted_until >= current_step
    "failed_expired_anchors": [...], # failed_approach_count > 0 或 strong_negative > 0.5
    "memory_reuse_hits": N,          # 可复用数量
}
```

Planner 在 `_best_object_anchor()` 中利用这些信息：

- `unfailed_anchors` → score +0.6（重复目标时）
- `failed_approach_count > 0` → score -0.5
- `strong_negative_evidence > 0` → score -0.50 * value
- `cross_goal_preserved` → score +0.2
- `repeated_goal_source` → score +0.3

### 11.3 记忆的置信度控制

长期记忆的保留 / 删除遵循置信度：

```text
高置信、任务相关、反复观测 → 长期保留（cross_goal_preserved + 高 confidence）
低置信、过期、无关 → 衰减 / 折叠 / 删除（prune_low_confidence）
错误目标 → strong_negative_evidence 降权（_scan_reuse 将其标记为失败候选）
远处 object → 折叠进 room summary（adaptive_granularity）
```

### 11.4 剩余缺口

以下缺口已于 Section 11 实现中修复：

| 缺口 | 修复方式 |
|------|---------|
| Blocked edges 未参与 planner 候选筛选 | ✅ `_best_frontier()` 中检查 `edge.traversability < 0.2 \|\| blocked_count > 3` 时 skip；`blocked_count > 1` 时 `sem_score -= blocked_count * 0.5` |
| Environment_object 会被 prune 删除 | ✅ `prune_low_confidence()` 中保护 `confidence >= 0.55` 或 `multi_view_count >= 2` 的 OBJECT（即使 task-relevance=0）；`decay_all_confidences()` 同样保护 |

仍存在的缺口：

| 缺口 | 影响 | 建议 |
|------|------|------|
| Frontiers 在 goal 切换时无显式保留 | 新目标不记得哪些方向未探索 | 可标记 high-value frontier 为 cross_goal_preserved |


## 12. Navigation 设计

### 12.1 整体架构

GOATAgent 内部模块化，不拆 multi-agent：

```python
class GoatAgent:
    goal_manager: GoalManager
    perception_manager: PerceptionManager
    memory_writer: MemoryWriter
    structure_planner: StructurePlanner
    navigation_planner: NavigationPlanner
    stop_verifier: StopVerifier
    local_servo: LocalVisualServo
```

Plan 执行 proposal-first 规划：

```text
plan():
  1. structure_planner.select()                         → StructureTarget
  2. navigation_planner.collect_goal_proposals()         → object/room/frontier proposals
  3. _collect_hypothesis_goal_proposals()                → weak CLIP/VLM proposals
  4. navigation_planner.score_goal_proposals()           → unified score
  5. navigation_planner.select_best_proposal()           → selected GoalProposal
  6. navigation_planner.proposal_to_nav_target()         → NavTarget for execution
```

### 12.2 Planner 输入

Planner 不从 VLM 直接输出行动，也不再直接选择裸 `ObjectNode / Frontier / Room`。
所有候选先统一包装成 `GoalProposal`：

| 候选类型 | 来源 | 评分字段 |
|---------|------|---------|
| confirmed object | `ObjectMemoryPool.get_all()` | confidence + task_score + attr/relation + reachability |
| room / landmark / structure | `StructurePlanner.select()` | room_prior + relation/task context |
| frontier | `_collect_frontier_proposals()` | frontier_value + hypothesis + room_prior + blocked_check |
| hypothesis | `HypothesisPool.get_active()` | weak confidence, capped, requires VLM |
| visited waypoint | `_farthest_visited()` | fallback only |

### 12.3 GoalProposal 管道

`NavigationPlanner` 包含完整的 proposal pipeline：

| 方法 | 作用 |
|------|------|
| `collect_goal_proposals()` | 从 ObjectMemoryPool / StructureLayer / Frontier 收集 proposals |
| `_collect_hypothesis_goal_proposals()` | 从 HypothesisPool 收集 weak proposals |
| `score_goal_proposals()` | 统一评分（task_score + attr/relation/reachability/confidence） |
| `select_best_proposal()` | 从 active / needs_verify / confirmed proposals 中选最高分 |
| `proposal_to_nav_target()` | 将 selected proposal 转为执行层 NavTarget |

GoalProposal 引用已有 node，不新建重复 goal 实例：

```python
GoalProposal(
    goal_id, candidate_node_id, candidate_type, target_position,
    score, source, can_stop, requires_verification
)
```

proposal 约束：

```text
object_memory proposal: 可以 approach，可以 stop
clip / vlm_weak proposal: 只能探索 / 触发 VLM，can_stop=False
frontier proposal: 只能探索，can_stop=False
room / structure proposal: 只能引导搜索，can_stop=False
```

### 12.4 proposal_score 公式

```text
proposal_score =
  semantic_score
  + task_score
  + attribute_score
  + relation_score
  + reachability_score
  + history_bonus
  + frontier_value
  - distance_cost
  - risk_penalty
  - negative_evidence
```

### 12.5 reachability 估算

因 navmesh probe 不是通用可用，采用简化距离代理：

```python
_compute_reachability(candidates, position):
    reach = [1.0] * len(candidates)           # 默认可达
    cost = [min(d / 20.0, 1.0) for d in dists] # 路径代价
```

## 13. Approach / Stop

### 13.1 状态机

NavPhase enum 定义 8 个导航阶段：

```text
GLOBAL_SEARCH → ROUTE_TO_STRUCTURE → ROUTE_TO_OBJECT_ANCHOR
                                                 ↓
                                          LOCAL_VISUAL_APPROACH
                                                 ↓
                                            STOP_VERIFY
                                                 ↓
                                               STOP
                    ↓
              RECOVERY
```

规则：

- GLOBAL_SEARCH 不能直接 STOP
- LOCAL_VISUAL_APPROACH 不能直接 STOP
- 只有 STOP_VERIFY → STOP

### 13.2 Approach 流程

```text
_target_output_for_node() → ApproachPoint
  → _resolve_object_route() → ROUTE_TO_OBJECT_ANCHOR
    → at_anchor_waypoint → LOCAL_VISUAL_APPROACH
      → Goal visible + range_bin in ("near", "close") + bbox >= min_approach
        → LocalVisualServo.act() → turn_left/right/move_forward
          → bbox_history + plateau + retreating check
            → ServoAction("hold") → STOP_VERIFY
```

### 13.3 3 层 Stop 验证

`StopVerifier.can_stop()` 的三层防线：

| 层 | 条件 | 说明 |
|----|------|------|
| layer1 | `confirm_buffer` 多数通过 | 多帧确认目标可见 |
| layer2 | bbox 增长 + approach 距离 + plateau creep | 接近且 bbox 非缩小 |
| layer3 | fresh VLM + stop_candidate + 居中 + visible + progress_ok + task_score>0.3 | 多角度验证 |

最终决定：`layer1 && layer2 && layer3 && !retreating`

### 13.4 早停预防

保守策略：

```text
near          → 继续 approach（不停）
very_near     → stop_candidate（可能停）
very_near + bbox_area >= bbox_min_stop + 多帧确认 + bbox 未缩小 → STOP
```

具体条件：

- `range_bin in ("very_near", "close")` 才进入 stop band
- `bbox_area >= bbox_min_stop (0.18)` 或 `range_stop`
- `forward_action_count >= 3 (min_forward_before_stop)`
- `confirm_count >= 2 (servo_entry_evidence)`
- `not retreating`（bbox history 非持续缩小）

### 13.5 看到目标但不停

`_maybe_enter_servo_near_anchor()` 在每步 plan() 中检查：

```text
距 object_anchor <= 2.0m
+ goal_visible 或 VLM 检测匹配物体
+ bbox_area >= 0.04 或 range_bin in ("near", "close")
→ 进入 LOCAL_VISUAL_APPROACH
## 14. 总结

ConfTopo 是一个以 GoalGraph 理解任务、以 CLIP/VLM 形成和确认感知线索、以置信度机制固化长期语义拓扑记忆，并在多目标导航中复用经验的动态认知地图系统。
