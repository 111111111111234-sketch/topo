import json
with open('data/logs/goat_topo/final_14scenes/5cdEh9F2hJL_multigoal.json') as f:
    d = json.load(f)
for idx in [20, 21, 30, 31, 57, 58]:
    s = d['steps'][idx]
    print('=== step index', idx, 'step', s.get('step'), 'task', s.get('task_index'))
    print('keys', sorted(s.keys()))
    for k in ['stop_debug', 'sticky_debug', 'navigation', 'mode', 'target_node_id', 'target_position', 'agent_action', 'low_action']:
        print(' ', k, ':', s.get(k))
