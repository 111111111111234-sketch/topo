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

print("=== SUMMARY ===")
print(f"total steps: {len(steps)}")
for ti in sorted(by_task):
    ts = by_task[ti]
    goal = ts[0]["goal"]["target_object"]
    lows = Counter(x.get("low_action") for x in ts)
    nav = Counter((x.get("navigation_debug") or {}).get("reason") for x in ts if (x.get("navigation_debug") or {}).get("reason"))
    stops = [x for x in ts if x.get("low_action") == "stop"]
    stop_reasons = Counter((x.get("stop_debug") or {}).get("reason") for x in ts if x.get("stop_debug"))
    gmd = [x.get("goal_min_distance") for x in ts if x.get("goal_min_distance") is not None]
    print(f"\n--- task {ti}: {goal} ---")
    print(" low:", dict(lows))
    print(" nav top:", nav.most_common(8))
    print(" stop_debug reasons (sample):", stop_reasons.most_common(5))
    print(f" stops: {len(stops)}, goal_min: {min(gmd):.2f}m" if gmd else " no gmd")
    # last step stop_debug if any with stop_allowed true
    near_stop = [x for x in ts if (x.get("stop_debug") or {}).get("stop_allowed")]
    if near_stop:
        print(" stop_allowed steps:", len(near_stop), "last:", near_stop[-1]["stop_debug"])
    else:
        # show most common blocking reason from last 20 steps
        last20 = [x.get("stop_debug", {}).get("reason") for x in ts[-20:] if x.get("stop_debug")]
        print(" last20 stop reasons:", Counter(last20).most_common(4))
