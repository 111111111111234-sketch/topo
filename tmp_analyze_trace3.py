import json
from collections import Counter, defaultdict
from pathlib import Path

p = Path("data/logs/goat_topo/final_14scenes/5cdEh9F2hJL_multigoal.json")
with open(p) as f:
    d = json.load(f)
steps = d["steps"]

by_task = defaultdict(list)
for s in steps:
    by_task[s["task_index"]].append(s)

for ti in sorted(by_task):
    ts = by_task[ti]
    goal = ts[0]["goal"]["target_object"]
    lows = Counter(x.get("low_action") for x in ts)
    plans = Counter(x.get("plan_action") for x in ts if x.get("plan_action"))
    nav = Counter((x.get("navigation_debug") or {}).get("reason") for x in ts if (x.get("navigation_debug") or {}).get("reason"))
    stops = [x for x in ts if x.get("low_action") == "stop"]
    rec = sum(1 for x in ts if (x.get("navigation_debug") or {}).get("reason") == "stuck_recovery")
    print(f"\n=== task {ti}: {goal} n={len(ts)} ===")
    print(" low:", dict(lows))
    print(" plan:", dict(plans))
    print(" nav top:", nav.most_common(6))
    print(" stuck_recovery steps:", rec)
    if stops:
        print(" STOP stop_debug:", stops[-1].get("stop_debug"))
    # min goal_min_distance trend
    gmd = [x.get("goal_min_distance") for x in ts if x.get("goal_min_distance") is not None]
    if gmd:
        print(f" goal_min_distance: start={gmd[0]:.2f} end={gmd[-1]:.2f} best={min(gmd):.2f}")
