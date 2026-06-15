# ConfTopo VLM 感知与 Multi-Agent 改造计划

## 1. 改造目标

当前 `ConfTopoGOATAgent` 已经具备完整导航闭环：

```text
observe -> update_memory -> plan -> act
```

但感知、记忆、规划、停止判断、目标重定位等逻辑都集中在 `goat_agent.py`（~2300 行）中。后续如果直接同时替换 VLM 和拆 multi-agent，风险会很高，容易出现闭环不稳、问题难定位、模块边界不清的问题。

本改造计划采用更稳的路线：

```text
先统一感知接口（PerceptionReport）
再抽出触发策略
再接入 VLM
再拆 MemoryAgent / PlannerAgent
最后引入 Blackboard + Orchestrator
```

最终目标是把 ConfTopo 从单体 agent 改造成：

```text
VLM 感知 -> 结构化长期拓扑记忆 -> 任务驱动图推理 -> 多模块协作式导航
```

也就是构建一个「会看、会记、会想、会走」的结构化导航框架。

---

## 2. 总体路线

```text
Phase A:  统一 PerceptionReport + 双轨感知策略
Phase B1: 抽出 perception trigger + fake VLM backend
Phase B2: 接入真实 VLM 感知后端
Phase C1: 抽出 MemoryAgent（waypoint + semantic nodes）
Phase C2: 抽出 MemoryAgent（frontier + heavy objects + maintenance）
Phase D1: 抽出 PlannerAgent（plan + sticky/block）
Phase D2: 抽出 PlannerAgent（should_stop + reground + stuck recovery）
Phase E:  引入 Blackboard + Orchestrator
Phase F:  接入 GOAT-Bench 官方评测 + 消融实验
```

原则：

- 不要一开始就拆成多个 agent，也不要第一版就删掉 CLIP / GroundingDINO。
- 每个子阶段结束后必须跑回归测试（`test_phase3.py` + `run_goat_multigoal_acceptance.py`），确认闭环不退化。

第一版最稳策略：

```text
Qwen3-VL-8B  触发式场景理解（room / objects / goal / portals）
CLIP         保留用于 visual embedding + 非触发步的粗分类
GroundingDINO 保留作 baseline / fallback（消融对比用）
DynamicTopoMap 保持不动
Navigator    保持外部
```

---

## 3. 当前代码耦合点

当前数据流：

```text
observe()
  ├─ rgb_embed -> LightPerceiver (CLIP 余弦相似度)
  │    → _cur_perception: {room_label, goal_scores, landmark_scores, best_goal_sim}
  └─ rgb 缓存，稍后按需触发 heavy perception

update_memory()
  ├─ _add_visited_waypoint()
  ├─ _add_semantic_nodes()       ← 读 _cur_perception
  ├─ _add_heavy_object_nodes()   ← 调 HeavyPerceiver, 写 _cur_heavy_observations
  ├─ _generate_frontiers()
  └─ decay / merge / prune / assign_waypoint_to_room

plan() / should_stop()
  └─ 读 topo_map + _cur_perception + heavy confirm
```

关键耦合点：

| 字段 | 类型 | 消费者 |
|------|------|--------|
| `_cur_perception` | `dict` (room_label, goal_scores, ...) | `_add_semantic_nodes`, `_should_run_heavy_perception`, `plan` |
| `_cur_heavy_observations` | `list[ObjectObservation]` | `_add_heavy_object_nodes`, `_room_clf` |
| `_cur_rgb` | raw image | `HeavyPerceiver.detect()` |
| `_cur_rgb_embed` | `np.ndarray` (CLIP) | `LightPerceiver.perceive()`, 节点 embedding |

改造分两层：

```text
换 VLM：  主要改 perception 层 + trigger
拆 agent：主要改 agent 编排层（goat_agent → orchestrator）
```

---

## 4. Phase A：统一感知接口 + 双轨策略

### 目标

把当前两套感知结果统一成一个结构化输出，并定义「每步」和「触发时」的双轨策略：

```text
每步（快，< 1ms）:
  CLIP embedding → PerceptionReport（仅 room/goal/landmark scores + visual_embed）
  objects 字段为空

触发时（慢，VLM 或 GroundingDINO）:
  完整 PerceptionReport（含 objects, scene_summary, portals, goal_visible 等）
```

后续 memory 和 planner 只读取 `PerceptionReport`，不再关心感知来源。

### 新增文件

```text
conftopo/perception/perception_report.py
```

### 核心结构

```python
@dataclass
class PerceptionReport:
    # --- 原 LightPerceiver 字段（每步都有） ---
    room_label: str = "unknown"
    room_confidence: float = 0.0
    room_scores: list[tuple[str, float]] = field(default_factory=list)
    goal_scores: list[tuple[str, float]] = field(default_factory=list)
    landmark_scores: list[tuple[str, float]] = field(default_factory=list)
    best_goal_sim: float = 0.0

    # --- 原 HeavyPerceiver 字段（仅触发时有） ---
    objects: list[ObjectObservation] = field(default_factory=list)

    # --- VLM 增强字段（仅触发时有） ---
    scene_summary: str = ""
    goal_visible: bool = False
    goal_reason: str = ""
    portals: list[str] = field(default_factory=list)
    uncertainty: float = 0.0

    # --- 兼容字段 ---
    visual_embed: Optional[np.ndarray] = None   # CLIP embedding，每步都有
    source: str = "none"                         # "clip", "clip_groundingdino", "vlm"
    is_full: bool = False                        # True = 触发式完整报告
    step_id: int = 0
    raw: dict = field(default_factory=dict)
```

### 改造原则

- `_cur_perception` 和 `_cur_heavy_observations` 统一替换为 `_cur_report: PerceptionReport`。
- `_add_semantic_nodes()` 读 `report.room_*` / `report.goal_scores` / `report.landmark_scores`。
- `_add_heavy_object_nodes()` 读 `report.objects`。
- 第一阶段不改变导航行为，只改变数据接口。
- 写一个 `ClipGdinoReportBuilder` 把 `LightPerceiver` + `HeavyPerceiver` 的输出组装成 `PerceptionReport`。

### 完成标准

- `backend="clip_groundingdino"` 路径行为不变。
- 现有 trace / visualization / test_phase3.py 不崩。
- memory / planner 不直接依赖具体感知后端。
- `PerceptionReport` 有 schema 级单测（字段类型、默认值、序列化）。

---

## 5. Phase B：抽出触发策略 + 接入 VLM

### Phase B1：触发策略独立化 + fake VLM

**目标**：把 `_should_run_heavy_perception()` 及其依赖的状态抽成独立模块，使 Phase C/D 拆 agent 时不必再动触发逻辑。

新增文件：

```text
conftopo/perception/perception_trigger.py
conftopo/perception/vlm_backend.py          # FakeVLMBackend 先行
```

`PerceptionTrigger` 提取自 `goat_agent._should_run_heavy_perception()`，封装为：

```python
class PerceptionTrigger:
    def should_run(self, step: int, goal_local_step: int,
                   best_goal_sim: float, position: np.ndarray,
                   topo_map: DynamicTopoMap, reground_state: str,
                   ...) -> tuple[bool, str]:
        ...
```

`FakeVLMBackend` 返回固定/随机的结构化 JSON，用于无 GPU 时跑通全链路。

**完成标准**：

- `goat_agent` 调用 `PerceptionTrigger.should_run()` 代替内联逻辑，行为不变。
- fake backend 能产出合法 `PerceptionReport`，multigoal acceptance 能跑。

### Phase B2：真实 VLM 接入

新增文件：

```text
conftopo/perception/vlm_perceiver.py    # VLM JSON → PerceptionReport
conftopo/perception/vlm_prompts.py      # 结构化 JSON prompt 模板
```

**VLM 选型：Qwen3-VL**

主选 Qwen3-VL 系列（Apache 2.0 开源，vLLM 原生支持）。Qwen3-VL 相比上代有三项关键升级：DeepStack 多层 ViT 特征融合（细粒度视觉）、Interleaved-MRoPE（空间位置感知）、Advanced Spatial Perception（3D grounding / 遮挡判断），特别适合室内导航场景的物体识别与空间推理。

可用型号及部署建议：

| 型号 | 架构 | 推理显存 (FP16) | 适用场景 |
|------|------|-----------------|----------|
| **Qwen3-VL-8B-Instruct** | Dense 8B | ~18 GB (单卡 A100/4090) | **默认首选**，精度/速度平衡 |
| Qwen3-VL-4B-Instruct | Dense 4B | ~10 GB | 显存紧张时降级 |
| Qwen3-VL-2B-Instruct | Dense 2B | ~6 GB | 极端低资源 / 快速原型 |
| Qwen3-VL-30B-A3B-Instruct | MoE 30B (激活 3B) | ~10 GB | MoE 推理优化后，速度接近 4B |
| Qwen3-VL-32B-Instruct | Dense 32B | ~70 GB (2×A100) | 更强能力，多卡环境 |

部署方式：

```bash
# vLLM 部署（推荐，vllm >= 0.11.0）
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype auto \
    --limit-mm-per-prompt image=1 \
    --max-model-len 4096

# 或 Ollama 本地（开发调试）
ollama run qwen3-vl:8b
```

**推荐配置**：

- **第一版默认**：`Qwen3-VL-8B-Instruct` (local vLLM)
- **低资源 fallback**：`Qwen3-VL-4B-Instruct` 或 `Qwen3-VL-30B-A3B-Instruct` (MoE)
- **Thinking 版**（`Qwen3-VL-8B-Thinking`）仅在 stop confirmation 等需要推理链的场景按需启用，常规感知不用
- 通过 `vlm_backend.py` 统一 OpenAI 兼容 API 接口，切换模型只改 endpoint + model name

**不选 API-only 方案的原因**：GOAT-Bench 每 episode 数百步触发式调用 VLM，API 延迟和成本不可控；local 部署可与 Habitat-Sim 同机运行，延迟可控在 0.3–1s/次。

**触发条件**（复用 `PerceptionTrigger`）：

```text
目标切换时（goal_warmup）
接近目标时（stop_confirmation_near_goal）
低置信度时（low_object_confidence）
卡住时（stuck 触发）
周期性 refresh（interval）
需要 stop confirm 时（reground scanning）
coarse summary context（进入新房间区域）
```

**VLM prompt 设计**：

Qwen3-VL 支持 OpenAI 兼容的 chat API，图片通过 base64 或 URL 传入。prompt 采用 system + user 两层结构，要求 JSON 输出：

```text
System:
  You are a navigation perception agent inside a house.
  Given the robot's egocentric RGB image and its current goal,
  output a JSON object describing the scene.
  Only output valid JSON, no explanation.

User:
  <image>
  Goal: find the sink
  Respond with this JSON schema:
  {
    "room": {"label": str, "confidence": float},
    "objects": [{"label": str, "region": str, "confidence": float}],
    "goal_visible": bool,
    "goal_reason": str,
    "portals": [str],
    "scene_summary": str,
    "uncertainty": float
  }
```

输出示例：

```json
{
  "room": {"label": "kitchen", "confidence": 0.85},
  "objects": [
    {"label": "sink", "region": "center-left", "confidence": 0.9},
    {"label": "cabinet", "region": "right", "confidence": 0.7}
  ],
  "goal_visible": true,
  "goal_reason": "sink is visible on the left side of the counter",
  "portals": ["door on the right leading to hallway"],
  "scene_summary": "A kitchen with sink on the left, cabinets on the right, door visible",
  "uncertainty": 0.1
}
```

**Qwen3-VL 特有能力利用**：

- **Spatial Perception**：Qwen3-VL 支持 2D/3D grounding，`region` 字段可直接利用其空间位置理解（center-left / far-right 等）
- **物体识别广度**：预训练覆盖室内常见物体，无需像 GroundingDINO 那样手动拼 caption
- **结构化输出**：配合 vLLM 的 `guided_decoding` 可强制 JSON schema 输出，减少解析失败

**完成标准**：

- `backend="clip_groundingdino"` 和 `backend="vlm"` 均可运行。
- VLM 超时、JSON 解析失败、空输出时 fallback 到上一帧 report 或 CLIP backend，系统不崩。
- fake backend 测试覆盖全部异常路径。

---

## 6. Phase C：抽出 MemoryAgent（分两步）

### Phase C1：waypoint + semantic nodes

**迁移的方法**（从 `goat_agent.py`）：

```text
_add_visited_waypoint()
_consume_reached_frontiers()
_add_semantic_nodes()
_record_view_context() / _record_view_context_smoothed()
_record_scene_vocabulary()
_link_semantic_to_room_summary()
_new_landmark_attributes() / _update_landmark_history()
_object_room_attributes() / _object_room_compatible() / _update_object_room_context()
_normalized_room_label()
```

### Phase C2：frontier + heavy objects + maintenance

**迁移的方法**：

```text
_generate_frontiers() / _generate_initial_frontiers()
_add_heavy_object_nodes()
_object_position_from_bbox()
_heavy_labels()

以及 update_memory() 中的维护逻辑:
  topo_map.decay_all_confidences()
  topo_map.merge_nearby_nodes()
  topo_map.adaptive_granularity()
  topo_map.prune_low_confidence()
  topo_map.assign_waypoint_to_room()
```

### 新增文件

```text
conftopo/agents/memory_agent.py
```

### 接口

```python
class MemoryAgent:
    def __init__(self, config: ConfTopoConfig):
        self.topo_map = DynamicTopoMap(config.memory)
        self._room_clf = RoomClassifier(...)

    def run(self, pose, heading, report: PerceptionReport,
            goal: GoalNode, prev_vp_id: str | None) -> str:
        """更新 topo_map，返回当前 vp_id。"""
        ...
```

### 边界约束

- MemoryAgent **不**调用 VLM / CLIP / GroundingDINO。
- MemoryAgent **不**做 plan / stop / 触发感知。
- MemoryAgent 输入是 pose + goal + `PerceptionReport` + nav_event。
- MemoryAgent 输出是更新后的 `DynamicTopoMap`（+ 当前 vp_id）。
- `DynamicTopoMap` 不拆成 agent，仍由 MemoryAgent 托管。

### 完成标准

- C1 结束后 `test_phase3.py` + multigoal acceptance 行为不退化。
- C2 结束后同上。
- `goat_agent.update_memory()` 简化为调用 `memory_agent.run()`。

---

## 7. Phase D：抽出 PlannerAgent（分两步）

### Phase D1：plan + sticky/block

**迁移的方法**：

```text
plan()                              # 核心
_candidate_skip_reason()
_candidate_anchored_skip_reason()
_select_structure_target()
_structure_node_has_semantic_match()
_sticky_plan_if_valid()
_clear_sticky()
_block_target() / _is_blocked_target() / _active_blocked_targets()
_consume_or_block_target()
_resolve_navigable_target()
_resolve_object_waypoint_anchor()
_target_output_for_node()
_apply_reachability_components()
_apply_structure_anchor_bonus()
_compute_reachability_components()
_debug_plan_state()
_is_stuck()
```

### Phase D2：should_stop + reground + recovery

**迁移的方法**：

```text
should_stop()
on_goal_reached()
_current_goal_bbox_confirmation()
_has_recent_goal_detection()
_is_goal_object_node()
_has_near_goal_object_node()
_heading_toward_object_ok()
_object_direct_nav_allowed()
_goal_object_memory_ok()
_object_center_dist()

_plan_regrounding()
_plan_to_reground_target()
_plan_local_reground_search()
_fail_regrounding()
_reset_reground_state()
_try_start_regrounding()
_reground_scan_action()
```

### 新增文件

```text
conftopo/agents/planner_agent.py
```

### 输出结构

```python
@dataclass
class PlanResult:
    plan_action: str                          # "navigate" / "stop" / "recover" / "reground" / "no_target"
    target_id: Optional[str] = None
    target_position: Optional[np.ndarray] = None
    is_exploration: bool = False
    should_stop: bool = False
    action: Optional[str] = None              # 离散动作（turn_left 等），仅 recover/reground 时
    scores: dict = field(default_factory=dict)
    structure_target_id: Optional[str] = None
    debug: dict = field(default_factory=dict)
```

### 边界约束

- PlannerAgent **不**写 memory（不调用 `topo_map.add_node` 等）。
- PlannerAgent **不**调用感知后端。
- PlannerAgent 输入是 goal + topo_map（只读） + PerceptionReport + nav_event。
- PlannerAgent 可以维护 sticky/block/reground 内部状态。

### 完成标准

- D1 结束后 multigoal acceptance 行为不退化。
- D2 结束后同上。
- `goat_agent.plan()` 和 `goat_agent.should_stop()` 简化为调用 `planner_agent`。

---

## 8. Phase E：Blackboard + Orchestrator

### 目标

`ConfTopoGOATAgent` 最终变成调度器（Orchestrator），不再包含业务逻辑。

### Agent 组成

```text
GoalAgent          目标管理（set_new_goal, instruction_graph）
PerceptionAgent    VLM / CLIP 感知 + 触发决策
MemoryAgent        DynamicTopoMap 读写
PlannerAgent       图推理 + 停止判断
Navigator          外部低层控制（不属于 Orchestrator 内部）
```

### Blackboard 结构

```python
@dataclass
class Blackboard:
    # 观测（每步由 Orchestrator 写入）
    rgb: Optional[np.ndarray] = None
    rgb_embed: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    heading: float = 0.0
    step_count: int = 0

    # GoalAgent 写入
    goal: Optional[GoalNode] = None
    instruction_graph: Optional[InstructionGraph] = None

    # PerceptionAgent 写入
    perception_report: Optional[PerceptionReport] = None

    # MemoryAgent 写入 / 托管
    topo_map: Optional[DynamicTopoMap] = None
    current_vp_id: Optional[str] = None

    # PlannerAgent 写入
    plan: Optional[PlanResult] = None

    # Navigator / Harness 写入
    nav_event: Optional[dict] = None
```

### 写权限约束

| 字段 | 写入者 | 其他 agent |
|------|--------|-----------|
| `perception_report` | PerceptionAgent | 只读 |
| `topo_map` | MemoryAgent | 只读 |
| `plan` | PlannerAgent | 只读 |
| `goal` / `instruction_graph` | GoalAgent | 只读 |
| `nav_event` | 外部 harness / Navigator | 只读 |

### 调度流程

```python
class Orchestrator:
    def step(self, obs: dict) -> dict:
        self.blackboard.update_obs(obs)

        self.goal_agent.run_if_needed(self.blackboard)
        self.perception_agent.run_if_triggered(self.blackboard)
        self.memory_agent.run(self.blackboard)
        self.planner_agent.run(self.blackboard)

        return self.blackboard.plan
```

### 新增文件

```text
conftopo/agents/blackboard.py
conftopo/agents/orchestrator.py
conftopo/agents/goal_agent.py
conftopo/agents/perception_agent.py   # 包装 PerceptionTrigger + VLM/CLIP backend
```

### 兼容策略

- `goat_agent.py` 保留为 thin wrapper，内部委托 `Orchestrator`。
- 外部 harness (`run_goat_multigoal_acceptance.py`) 调用接口不变。
- 配置 `agent.mode: "monolithic" | "multi_agent"` 控制走老路径还是新 Orchestrator。

### 完成标准

- `mode="monolithic"` 行为等价于改造前。
- `mode="multi_agent"` multigoal acceptance 结果相同。
- agent 之间不直接互调，全部经 Blackboard。

---

## 9. Phase F：GOAT-Bench 官方评测

### 接入路线（分两步）

**注意**：GOAT-Bench 官方不是 `reset/act/update_goal` 式 Modular Agent，而是 **Habitat `NetPolicy` + PPO trainer eval loop**。

#### Step 1（Path B，建议先做）

用官方 `Goat-v1` task + `GoatSuccess` / `GoatSPL` measurement，自写 eval loop 调 Orchestrator：

```text
官方 Habitat env (Goat-v1)
+ 官方 GoatSuccess / GoatSPL measurement
+ 自写 eval harness 调用 Orchestrator.step()
+ 官方 subtask_stop 动作语义
```

这等于把现有 `run_goat_multigoal_acceptance.py` **升级**为使用官方 env + measurement，但 agent 循环仍由自己控制。

关键对齐项：

| 项 | 官方标准 | 需要确认 |
|----|---------|---------|
| 每 subtask 步数 | 500 | harness 一致 |
| 成功距离 | < 1m euclidean | SPL 计算一致 |
| 动作空间 | 6 维（含 look up/down） | 是否用到 |
| goal 模态 | object / language / image | adapter 覆盖 |
| subtask 切换 | `subtask_stop` action | 映射到 `set_new_goal` |

#### Step 2（Path A，可选但更标准）

注册 `ConfTopoGOATPolicy` 为 Habitat `NetPolicy`，直接用 `goat_bench.run --run-type eval`：

```python
@baseline_registry.register_policy(name="ConfTopoGOATPolicy")
class ConfTopoGOATPolicy(NetPolicy):
    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False):
        # observations → PerceptionReport → Orchestrator → PlanResult → 离散动作
        ...
```

### ConfTopo 只负责

```text
perception / memory / planning / reasoning / target selection
```

### 官方继续负责

```text
dataset / split / episode / env / action loop / metric / benchmark output
```

不要重写 GOAT-Bench metric，也不要用自定义 trace runner 当最终 benchmark 结果。

### 完成标准

- Path B: 使用官方 `GoatSuccess` + `GoatSPL` 得到可与 SenseAct-NN 直接对比的数字。
- 现有 `scripts/run_goat_*` 保留作 debug，不作为论文数字来源。

---

## 10. 实验与消融

### 消融表

| 配置 | 说明 |
|------|------|
| CLIP + GroundingDINO (baseline) | 原始感知后端 |
| **Qwen3-VL-8B perception** | 替换为 VLM 感知 |
| Qwen3-VL-4B / 30B-A3B | VLM 规模消融 |
| Monolithic goat_agent | 单体 agent |
| Multi-agent Orchestrator | 拆分后 |
| w/o structured memory | 不写 DynamicTopoMap，仅当前帧 |
| w/o two-stage planner | 单阶段选点 |
| w/o VLM stop confirmation | 去掉 VLM 辅助 stop 判断 |
| w/o confidence gating | 去掉置信度门控 |
| w/o cross-goal memory reuse | 每个 goal reset memory |

### 主表指标

在 GOAT-Bench 上（和 SenseAct-NN 对比）：

| 指标 | 说明 |
|------|------|
| SR (subtask) | 每个 subtask 的成功率 |
| SPL (subtask) | 每个 subtask 的 SPL |
| Composite SR | 所有 subtask 全部成功的 episode 比例 |
| Memory Reuse Rate | 后续 goal 命中已有记忆的比例 |
| Task 2/3/4 vs Task 1 | 后续目标效率提升 |

### 论文叙事

```text
PerceptionAgent (Qwen3-VL) 负责看
MemoryAgent (DynamicTopoMap) 负责记
PlannerAgent (GraphRetrieval) 负责想
Navigator (PointNav) 负责走
```

Navigator 不是主要创新点，低层控制使用官方 baseline 或已有 PathfinderExecutor。

---

## 11. 风险与默认策略

### VLM 延迟

| 风险 | 策略 |
|------|------|
| 每步调 VLM 会超时（GOAT 500 步/subtask） | 触发式调用，复用 `PerceptionTrigger` |
| Qwen3-VL-8B 单次推理 0.3–1s | 可接受；缓存结果到 waypoint，near goal / stuck 时才高频 |
| 显存不足（需 ~18GB） | 降级到 Qwen3-VL-4B (~10GB) 或 MoE 30B-A3B (~10GB) |
| 离线环境无网络 | 全部用 local vLLM 部署，不依赖外部 API |

### VLM 输出不稳定

| 风险 | 策略 |
|------|------|
| JSON 解析失败 | vLLM `guided_decoding` 强制 JSON schema；解析失败时 fallback 到上一帧 report |
| 幻觉物体 | 低置信结果不写入 topo map；多帧一致性确认 |
| 置信度尺度不可比 | VLM 自报 confidence + 归一化；与 CLIP 分数分开通道使用 |
| VLM server 崩溃 / 不可用 | `backend="clip_groundingdino"` 全链路仍可跑 |

### 架构过早拆分

| 风险 | 策略 |
|------|------|
| 闭环不稳 | Phase C/D 各分两步，每步跑回归 |
| debug 困难 | Blackboard 可全量 dump，每 agent 有 debug dict |
| 模块职责不清 | 写权限表严格执行（见 Phase E） |
| 性能退化 | `mode="monolithic"` 保留，A/B 对比 |

---

## 12. 测试与回归策略

### 每 Phase 必跑

```text
pytest conftopo/tests/test_phase3.py            # 单元 + 集成
python scripts/run_goat_multigoal_acceptance.py  # 多目标闭环
```

### Phase A 新增

```text
PerceptionReport schema 单测（字段类型、默认值、to_dict/from_dict）
ClipGdinoReportBuilder 单测（输出等价于原 _cur_perception + _cur_heavy_observations）
```

### Phase B 新增

```text
FakeVLMBackend 单测（覆盖：正常 JSON / 空输出 / 超时 / 格式错误）
PerceptionTrigger 单测（各触发条件）
VLM → PerceptionReport 端到端 smoke test
```

### Phase C/D 新增

```text
MemoryAgent 单测（给定 report + pose → 验证 topo_map 状态）
PlannerAgent 单测（给定 topo_map + goal → 验证 PlanResult）
Orchestrator 集成测试（等价于原 goat_agent.step）
```

### 回归判据

```text
multigoal acceptance 的 SR / SPL / 步数不退化（delta < 1%）
topo_map 节点数 / 类型分布与改造前一致
plan 输出的 target_id 序列与改造前一致（mode=monolithic 时）
```

---

## 13. goat_agent.py 方法迁移清单

### → MemoryAgent

| 方法 | 子阶段 |
|------|--------|
| `_add_visited_waypoint()` | C1 |
| `_consume_reached_frontiers()` | C1 |
| `_add_semantic_nodes()` | C1 |
| `_record_view_context()` / `_record_view_context_smoothed()` | C1 |
| `_record_scene_vocabulary()` | C1 |
| `_link_semantic_to_room_summary()` | C1 |
| `_new_landmark_attributes()` / `_update_landmark_history()` | C1 |
| `_object_room_attributes()` / `_object_room_compatible()` / `_update_object_room_context()` | C1 |
| `_normalized_room_label()` | C1 |
| `_generate_frontiers()` / `_generate_initial_frontiers()` | C2 |
| `_add_heavy_object_nodes()` | C2 |
| `_object_position_from_bbox()` | C2 |
| `_heavy_labels()` | C2 |
| `update_memory()` 中的 decay/merge/prune/assign | C2 |

### → PlannerAgent

| 方法 | 子阶段 |
|------|--------|
| `plan()` | D1 |
| `_candidate_skip_reason()` / `_candidate_anchored_skip_reason()` | D1 |
| `_select_structure_target()` / `_structure_node_has_semantic_match()` | D1 |
| `_sticky_plan_if_valid()` / `_clear_sticky()` | D1 |
| `_block_target()` / `_is_blocked_target()` / `_active_blocked_targets()` | D1 |
| `_consume_or_block_target()` | D1 |
| `_resolve_navigable_target()` / `_resolve_object_waypoint_anchor()` | D1 |
| `_target_output_for_node()` | D1 |
| `_apply_reachability_components()` / `_apply_structure_anchor_bonus()` | D1 |
| `_compute_reachability_components()` | D1 |
| `_debug_plan_state()` / `_is_stuck()` | D1 |
| `should_stop()` | D2 |
| `on_goal_reached()` | D2 |
| `_current_goal_bbox_confirmation()` / `_has_recent_goal_detection()` | D2 |
| `_is_goal_object_node()` / `_has_near_goal_object_node()` | D2 |
| `_heading_toward_object_ok()` / `_object_direct_nav_allowed()` | D2 |
| `_goal_object_memory_ok()` / `_object_center_dist()` | D2 |
| `_plan_regrounding()` / `_plan_to_reground_target()` / `_plan_local_reground_search()` | D2 |
| `_fail_regrounding()` / `_reset_reground_state()` / `_try_start_regrounding()` / `_reground_scan_action()` | D2 |

### → GoalAgent

| 方法 | 阶段 |
|------|------|
| `set_new_goal()` | E |
| `set_environment_landmark_labels()` | E |
| `reset()` / `reset_keep_memory()` | E |

### → PerceptionAgent

| 方法 | 阶段 |
|------|------|
| `observe()` | E（Phase B 先抽 trigger） |
| `_should_run_heavy_perception()` | B1 → `PerceptionTrigger` |
| `set_heavy_perceiver()` | E |

### 保留在 Orchestrator / goat_agent

| 方法 | 说明 |
|------|------|
| `step()` | 变为 Orchestrator 调度 |
| `act()` | 组装最终输出 |
| `on_navigation_event()` | 转发到 PlannerAgent |
| `memory_stats()` | 聚合各 agent 统计 |

---

## 14. 推荐落地顺序

```text
 1.  [Phase A]  新增 PerceptionReport schema + 单测
 2.  [Phase A]  写 ClipGdinoReportBuilder，goat_agent 兼容读取
 3.  [Phase A]  跑回归，确认行为不变
 4.  [Phase B1] 抽出 PerceptionTrigger + FakeVLMBackend
 5.  [Phase B1] 跑回归
 6.  [Phase B2] 接真实 VLM（local 小模型优先）
 7.  [Phase B2] 跑回归，对比 CLIP vs VLM
 8.  [Phase C1] 迁移 waypoint + semantic nodes → MemoryAgent
 9.  [Phase C1] 跑回归
10.  [Phase C2] 迁移 frontier + heavy + maintenance → MemoryAgent
11.  [Phase C2] 跑回归
12.  [Phase D1] 迁移 plan + sticky/block → PlannerAgent
13.  [Phase D1] 跑回归
14.  [Phase D2] 迁移 should_stop + reground → PlannerAgent
15.  [Phase D2] 跑回归
16.  [Phase E]  Blackboard + Orchestrator + GoalAgent + PerceptionAgent
17.  [Phase E]  monolithic vs multi-agent A/B 对比
18.  [Phase F]  接 GOAT-Bench 官方 env + measurement (Path B)
19.  [Phase F]  可选 register_policy (Path A)
20.  [Phase F]  消融实验 + 论文表格
```

---

## 15. 目录结构（最终态）

```text
conftopo/
├── core/                                # 不动
│   ├── dynamic_topo_map.py
│   ├── confidence.py
│   ├── instruction_graph.py
│   ├── rule_scorer.py
│   ├── room_classifier.py
│   └── ...
├── perception/
│   ├── perception_report.py             # [Phase A] 统一 schema
│   ├── perception_trigger.py            # [Phase B1] 触发策略
│   ├── clip_gdino_report_builder.py     # [Phase A] CLIP+GDINO → Report
│   ├── vlm_perceiver.py                 # [Phase B2] VLM → Report
│   ├── vlm_backend.py                   # [Phase B1] fake + real backend
│   ├── vlm_prompts.py                   # [Phase B2] prompt 模板
│   ├── light_perceiver.py               # 保留
│   ├── heavy_perceiver.py               # 保留
│   └── clip_runtime.py                  # 保留
├── agents/
│   ├── blackboard.py                    # [Phase E]
│   ├── orchestrator.py                  # [Phase E]
│   ├── goal_agent.py                    # [Phase E]
│   ├── perception_agent.py              # [Phase E]
│   ├── memory_agent.py                  # [Phase C]
│   ├── planner_agent.py                 # [Phase D]
│   ├── base_agent.py                    # 保留
│   └── goat_agent.py                    # 逐步瘦身 → thin wrapper
├── navigation/
│   └── pathfinder_executor.py           # 保留，Navigator 外部
├── adapters/
│   ├── goat_official_policy.py          # [Phase F] Path A
│   ├── goat_eval_harness.py             # [Phase F] Path B
│   ├── etpnav_adapter.py               # 已有
│   └── soon_adapter.py                 # 已有
└── config.py                            # 新增 perception.backend / agent.mode
```

---

## 16. 一句话总结

本次改造不是简单拆 `goat_agent.py`，也不是直接把 CLIP/GDINO 换成 VLM，而是按 **统一接口 → 抽触发 → 接 VLM → 拆记忆 → 拆规划 → 编排 → 官方评测** 的稳定顺序，逐步将 ConfTopo 从单体 agent 改造成多模块协作导航框架。

最终目标：

```text
PerceptionAgent (Qwen3-VL-8B)        会看
MemoryAgent (DynamicTopoMap)          会记
PlannerAgent (GraphRetrieval)         会想
Navigator (PointNav / Pathfinder)     会走
```

每个阶段有回归测试，每个模块可独立 ablation，最终在 GOAT-Bench 官方 metric 上出论文数字。
