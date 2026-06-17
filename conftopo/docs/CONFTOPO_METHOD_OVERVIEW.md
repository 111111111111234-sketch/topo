# ConfTopo 方法思路整理

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

## 4. 记忆：DynamicTopoMap

ConfTopo 的长期记忆不是纯文本历史，也不是稠密语义地图，而是动态拓扑图：

```text
DynamicTopoMap = nodes + edges + confidence + temporal update
```

主要节点：


| 节点                 | 含义          |
| ------------------ | ----------- |
| WAYPOINT_VISITED   | agent 经过的位置 |
| WAYPOINT_FRONTIER  | 可能值得探索的方向   |
| WAYPOINT_CANDIDATE | 候选导航点       |
| OBJECT             | 物体语义节点      |
| LANDMARK           | 门、走廊、结构性锚点等 |
| ROOM               | 房间或区域摘要     |


主要边：


| 边            | 含义                      |
| ------------ | ----------------------- |
| NAVIGABLE    | 两个 waypoint 可通行         |
| OBSERVED_AT  | 某个位置观测到某个物体             |
| BELONGS_TO   | 物体 / waypoint 属于某个 room |
| VISIBLE_FROM | 某个 landmark 可从某个位置看到    |


记忆更新的目标不是“记录所有东西”，而是形成可规划的结构：

```text
哪里走过
哪里没探索
哪里看到过目标相关物体
哪些区域可能属于厨房 / 卧室 / 走廊
哪些门或通道可以作为结构锚点
```

## 5. 置信度与记忆维护

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
结构锚点 bonus
```

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

## 7. 多目标记忆复用

GOAT-Bench 的关键是 multi-goal episode。ConfTopo 在 goal 切换时：

```text
保留 DynamicTopoMap
保留已探索结构
保留 object / room / landmark memory
清空当前 goal 的短期规划状态
重新设置 goal labels / landmarks / room prior
```

也就是：

```text
episode reset -> 清空记忆
goal switch -> 不清空记忆
```

这样前一个目标探索到的信息可以为后一个目标服务。例如：

```text
Goal 1: find sofa
探索过程中建立 living room / hallway / doorway 结构

Goal 2: find tv
planner 可直接利用 living room memory，而不是重新随机探索
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