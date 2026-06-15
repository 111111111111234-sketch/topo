import json
import numpy as np
with open('data/logs/goat_topo/final_14scenes/5cdEh9F2hJL_multigoal.json') as f:
    d = json.load(f)
# task0 stop step
s = d['steps'][21]
pos = np.array(s['position'])
print('task0 stop pos', pos)
# find object nodes in topo at that step
topo = s.get('topo', {})
nodes = topo.get('nodes', [])
for n in nodes:
    if n.get('type') == 'object' and 'display' in (n.get('label') or '').lower():
        p = np.array(n['position'])
        attrs = n.get('attributes', {})
        ap = attrs.get('best_approach_position')
        d_obj = np.linalg.norm(p - pos)
        d_ap = np.linalg.norm(np.array(ap) - pos) if ap else None
        print('object', n.get('id'), n.get('label'))
        print('  obj dist', d_obj, 'approach dist', d_ap, 'approach', ap)
