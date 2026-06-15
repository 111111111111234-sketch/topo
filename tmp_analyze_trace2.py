import json
from collections import defaultdict
from pathlib import Path

p = Path("data/logs/goat_topo/final_14scenes/5cdEh9F2hJL_multigoal.json")
with open(p) as f:
    d = json.load(f)
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
    plan_actions = [x.get("plan_action") for x in ts if x.get("plan_action")]
    nav_reasons = []
    for x in ts:
        nav = x.get("navigation_debug") or {}
        if isinstance(nav, dict) and nav.get("reason"):
            nav_reasons.append(nav["reason"])
    print(f"\n=== task {ti}: {goal} steps={len(ts)} ===")
    print(" low_actions:", {a: lows.count(a) for a in sorted(set(lows))})
    print(" agent_actions:", {a: agents.count(a) for a in sorted(set(agents))})
    if modes:
        print(" modes:", {m: modes.count(m) for m in sorted(set(modes))})
    if plan_actions:
        print(" plan_actions:", {m: plan_actions.count(m) for m in sorted(set(plan_actions))})
    if nav_reasons:
        from collections import Counter
        c = Counter(nav_reasons)
        print(" nav reasons top:", c.most_common(8))
    stops = [x for x in ts if x.get("low_action") == "stop"]
    if stops:
        sd = stops[-1].get("stop_debug")
        print(" stop_debug:", sd)
    # stuck recovery / no_target
    rec = [x for x in ts if x.get("mode") in ("stuck_recovery", "no_target_fallback")]
    print(" recovery steps:", len(rec))
    # sample last 5
    for s in ts[-5:]:
        nav = s.get("navigation_debug") or {}
        print(
            f"  step {s['step']}: low={s.get('low_action')} agent={s.get('agent_action')} "
            f"plan={s.get('plan_action')} mode={s.get('mode')} tgt={s.get('target_node_id')} "
            f"nav={nav.get('reason') if isinstance(nav, dict) else None}"
        )
