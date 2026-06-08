# Phase 2 验收归档

> 归档日期：2026-06-03  
> 环境：`conda activate goat`  
> 主脚本：`scripts/run_goat_multigoal_acceptance.py`

## Checklist

| 项 | 状态 | 证据 |
|----|------|------|
| GOAT 单目标闭环 + pathfinder acceptance | **passed** | `data/logs/goat_topo/phase2_pathfinder_acceptance/` — object `GLAQ4DNUx5U` ep0（80 steps, `object_nodes=1`）；landmark `HY1NcmCgn3n` ep0（31 steps, `landmark_nodes=3`） |
| GOAT 多目标 memory preservation | **passed** | 4/4 task `memory_preserved=true`，节点 0→13→14→14→15 |
| GOAT 多目标 semantic memory reuse | **passed** | `multigoal_acceptance_report.json` — `memory_reuse_passed=true`，`semantic_reuse_passed=true`，task3 repeated goal 命中 task0 的 `obj_` 节点（9 hits） |
| SOON GoalNode 接口 | **passed (skeleton)** | `conftopo/adapters/soon_adapter.py` + 单测 |
| ETPNav alpha=0 bias 退化 | **passed (unit test)** | `conftopo/tests/test_phase2.py` |
| R2R-CE 完整 eval 对齐 | **deferred** | Phase 2 可选；见 `ETPNav/scripts/run_alpha_sensitivity.sh` |

**Phase 2 总体结论：`overall_passed=true`（multigoal acceptance）**

## Multi-goal 验收报告

路径：`data/logs/goat_topo/multigoal_acceptance/multigoal_acceptance_report.json`

| 字段 | 值 |
|------|-----|
| `memory_preservation_passed` | true |
| `semantic_build_passed` | true（`objects=1`） |
| `memory_reuse_passed` | true（240 waypoint reuse hits） |
| `semantic_reuse_passed` | true（22 hits） |
| `navigation_stable` | true（`collision_like_count=0`） |
| `overall_passed` | **true** |

### 运行命令

```bash
cd /workspace/tangyx7@xiaopeng.com
conda activate goat

python scripts/run_goat_multigoal_acceptance.py \
  --split val_seen \
  --scene GLAQ4DNUx5U \
  --episode-index 0 \
  --steps-per-goal 80 \
  --max-goals 4 \
  --viz
```

**关键配置**：CLIP 阈值从 Phase2 pathfinder acceptance 自动加载（`--phase2-summary data/logs/goat_topo/phase2_pathfinder_acceptance/summary.json`），与 object acceptance 对齐（`object_threshold≈0.05`）。默认 `steps-per-goal=80`。

### 产物

| 文件 | 说明 |
|------|------|
| `topo_trace_multigoal.json` | 320 step 完整 trace（含 topo snapshot，供 viz） |
| `multigoal_acceptance_report.json` | 验收判定与 task 摘要 |
| `viz/topo_map_final.png` | BEV 最终拓扑图 |
| `viz/goat_semantic_dual_view.mp4` | 双视角视频 |
| `viz/topo_map_semantic_growth.mp4` | 拓扑增长动画 |

## 已知限制（留到 Phase 2 之后）

- 无独立 `GOATConfTopoAdapter` 类（当前 `ConfTopoGOATAgent` + scripts 已够用）
- 视觉/回溯 frontier 未实现
- GOAT 官方 SR/SPL 未接入（需 GOAT task / success 判定）
- Docker part1→4 推送与 Phase 2 验收无依赖

## 下一步可选方向

1. **论文/实验向**：小规模 GOAT episode batch + trace 统计 — 见 `scripts/run_goat_batch_experiment.py`，结果 `data/logs/goat_topo/batch_experiment/batch_summary.json`
2. **工程向**：Docker part2→4 + `etpnav` 环境
3. **R2R 可选**：`ETPNav/scripts/run_alpha_sensitivity.sh` 跑 alpha=0 对齐
