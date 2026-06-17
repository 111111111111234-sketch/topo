# VLM 粗空间语义锚点

## 为什么这样做

原来的 VLM object 写图方式接近：

```text
bbox size -> 粗估 depth -> object_position -> 直接导航
```

这个假设不稳定。bbox 大小受物体真实尺寸、遮挡、裁剪、视角、相机参数影响很大。比如 sofa 的 bbox 大不一定近，cup 的 bbox 小不一定远，cabinet 只露出一部分时 bbox 也可能很小。

所以 VLM 路径不再把 bbox 大小当作主要深度来源。更合理的定位方式是：

```text
VLM 负责粗空间判断
bbox 负责视觉 grounding
TopoMap 负责保存语义锚点
近距离再做视觉确认
```

也就是不假装知道精确 3D object center，而是先记住“目标大概从哪个 waypoint 看见、在哪个方向、粗略远近、和哪些环境元素有关”。

## 做了什么

### 1. 扩展 VLM 输出

VLM object 现在支持这些粗空间字段：

```json
{
  "label": "sink",
  "visible": true,
  "visibility": "partially_visible",
  "bearing": "right_front",
  "range": "mid",
  "relation": ["beside cabinet", "near wall"],
  "room_context": "kitchen_or_bathroom",
  "confidence": 0.72,
  "bbox": [0.48, 0.32, 0.78, 0.70]
}
```

其中：

```text
bearing: left / center / right / left_front / right_front / front / unknown
range: near / mid / far / unknown
visibility: visible / partially_visible / not_visible / unknown
```

旧格式 VLM JSON 仍可解析；缺失字段会用安全默认值。

### 2. 扩展 ObjectObservation

`ObjectObservation` 增加并序列化这些字段：

```text
visible
visibility
bearing
range_bin
spatial_relation
room_context
```

这样 trace、report、测试和后续 debug 都能看到 VLM 的粗空间语义。

### 3. VLM object 写入 TopoMap 时改成 anchor

对 `source == "vlm"` 的 object：

```text
不再使用 bbox 大小估计 depth
不再把估出来的点当作真实 object center
position 默认使用当前 waypoint
anchor_waypoint_id 指向当前 waypoint
best_approach_position 指向 anchor waypoint
```

TopoMap node 会保存：

```text
anchor_waypoint_id
observed_from
bearing
range_bin
visibility
spatial_relation
vlm_room_context
position_source = "anchor_waypoint"
```

GroundingDINO 路径暂时保留旧的 bbox heuristic，避免影响已有 baseline。

### 4. planner 优先导航到 VLM anchor

如果 object 来自 VLM，或 `position_source == "anchor_waypoint"`：

```text
planner 不直接导航到伪 object_position
而是优先导航到 anchor_waypoint_id 对应的 waypoint
```

`range_bin` 的作用是指导后续行为：

```text
far: 先回到 anchor 或继续探索相关区域
mid: 在 anchor 附近寻找 frontier / candidate
near: 进入局部确认和 stop 判断
unknown: 回退到 room prior / landmark / frontier
```

### 5. stop 改为当前帧 VLM 确认

VLM 路径下，stop 不再主要依赖：

```text
object_dist <= threshold
```

而是优先看：

```text
当前帧 VLM 是否看到目标
visibility 是否可见或部分可见
range_bin 是否为 near
bbox 是否提供足够 grounding 证据
TopoMap 是否已有匹配目标记忆
```

bbox area 仍保留，但只作为“目标在当前画面中足够明显”的证据，不再当作 depth。

## 有什么作用

这次改造让 VLM 记忆更符合 RGB-only 导航的实际能力：

```text
远处靠语义锚点
中距离靠 frontier / anchor 附近探索
近距离靠当前视觉确认
```

它避免了伪 3D object center 带来的错误导航，同时保留 bbox 的 grounding 价值。后续如果接入真实 depth 或 monocular depth，可以在 bbox / mask 区域取深度，再把 object node 从“粗语义锚点”升级成“带真实几何的 object node”。

一句话：

```text
Qwen3-VL 提供粗空间语义，bbox 提供视觉证据，TopoMap 保存 waypoint 级语义锚点，planner 先找目标区域，最后近距离确认 stop。
```
