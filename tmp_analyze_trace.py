import json
from collections import defaultdict
from pathlib import Path

p = Path("data/logs/goat_topo/final_14scenes/5cdEh9F2hJL_multigoal.json")
with open(p) as f:
    d = json.load(f)
print("top keys:", list(d.keys()))
steps = d["steps"]
print("num steps", len(steps))

by_task = defaultdict(list)
for s in steps:
    by_task[s.get("task_index", -1)].append(s)

for ti in sorted(by_task):
    ts = by_task[ti]
    goal = ts[0].get("goal", {}).get("target_object")
    lows = [x.get("low_action") for x in ts]
    agents = [x.get("agent_action") for x in ts]
    modes = [x.get("mode") for x in ts if x.get("mode")]
    stop = ts[-1]
    print(f"\n=== task {ti}: {goal} steps={len(ts)} ===")
    print(" low_actions:", {a: lows.count(a) for a in sorted(set(lows))})
    print(" agent_actions:", {a: agents.count(a) for a in sorted(set(agents))})
    if modes:
        print(" modes:", {m: modes.count(m) for m in sorted(set(modes))})
    print(" last step:", stop.get("step"), "low", stop.get("low_action"), "agent", stop.get("agent_action"))
    print(" stop_debug:", stop.get("stop_debug"))
    print(" mode:", stop.get("mode"))
    for s in ts[:3] + ts[-3:]:
        nav = s.get("navigation") or {}
        reason = nav.get("reason") if isinstance(nav, dict) else None
        print(
            f"  step {s['step']}: low={s.get('low_action')} agent={s.get('agent_action')} "
            f"tgt={s.get('target_node_id')} mode={s.get('mode')} nav={reason}"
        )
