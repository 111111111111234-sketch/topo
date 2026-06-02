from __future__ import annotations
import argparse, json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    'waypoint_visited': '#2563eb',
    'waypoint_frontier': '#f97316',
    'object': '#16a34a',
    'room': '#dc2626',
    'landmark': '#9333ea',
}
MARKERS = {
    'waypoint_visited': 'o',
    'waypoint_frontier': '^',
    'object': 'D',
    'room': 's',
    'landmark': 'P',
}
LABELS = {
    'waypoint_visited': 'visited waypoint',
    'waypoint_frontier': 'frontier',
    'object': 'object',
    'room': 'room',
    'landmark': 'landmark',
}


def collect_limits(steps):
    xs, zs = [], []
    for st in steps:
        p = st['position']
        xs.append(p[0]); zs.append(p[2])
        if st.get('target_position'):
            t = st['target_position']
            xs.append(t[0]); zs.append(t[2])
        for n in st['topo']['nodes']:
            xs.append(n['position'][0]); zs.append(n['position'][2])
    return (min(xs)-1.0, max(xs)+1.0), (min(zs)-1.0, max(zs)+1.0)


def fmt_scores(title, scores):
    lines = [title]
    if not scores:
        return [title + ': n/a']
    for item in scores[:3]:
        lines.append('  {}: {:.3f}'.format(item.get('label', 'n/a'), item.get('score', 0.0)))
    return lines


def draw_step(trace, idx, xlim, zlim, out_path=None):
    st = trace['steps'][idx]
    nodes = {n['id']: n for n in st['topo']['nodes']}
    fig, (ax, info) = plt.subplots(1, 2, figsize=(13, 6), dpi=130, gridspec_kw={'width_ratios': [2.1, 1]})

    for e in st['topo']['edges']:
        a = nodes.get(e['source'])
        b = nodes.get(e['target'])
        if not a or not b:
            continue
        nav = e.get('type') == 'navigable'
        ax.plot([a['position'][0], b['position'][0]], [a['position'][2], b['position'][2]],
                color='#6b7280', linewidth=1.2 if nav else 0.8,
                linestyle='-' if nav else '--', alpha=0.65, zorder=1)

    for typ in ['waypoint_visited', 'waypoint_frontier', 'object', 'room', 'landmark']:
        group = [n for n in nodes.values() if n['type'] == typ]
        if not group:
            continue
        ax.scatter([n['position'][0] for n in group], [n['position'][2] for n in group],
                   s=[70 + 180 * float(n.get('confidence', 0.5)) for n in group],
                   marker=MARKERS[typ], color=COLORS[typ], edgecolors='black',
                   linewidths=0.6, alpha=0.9, label=LABELS[typ], zorder=3)
        for n in group:
            label = n['id'] if not n.get('label') else '{}:{}'.format(n['id'], n['label'])
            ax.text(n['position'][0], n['position'][2] + 0.08, label, fontsize=7, ha='center')

    path = np.array([s['position'] for s in trace['steps'][:idx+1]])
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 2], color='#111827', linewidth=1.6, alpha=0.8, label='agent path', zorder=2)
    cur = st['position']
    ax.scatter([cur[0]], [cur[2]], marker='*', s=260, color='#facc15', edgecolors='black', linewidths=1.0, label='agent', zorder=5)
    if st.get('target_position'):
        tgt = st['target_position']
        ax.scatter([tgt[0]], [tgt[2]], marker='X', s=150, color='#ef4444', edgecolors='black', linewidths=0.8, label='planned target', zorder=4)
        ax.plot([cur[0], tgt[0]], [cur[2], tgt[2]], color='#ef4444', linewidth=1.0, linestyle=':', alpha=0.7, zorder=1)

    ax.set_title('DynamicTopoMap during GOAT exploration | step {}'.format(st.get('step')))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.22)
    ax.legend(loc='upper right', fontsize=7)

    mem = st['memory']
    goal = trace.get('current_goal', {})
    perception = st.get('perception', {})
    tasks = trace.get('tasks', [])[:4]
    task_text = '\n'.join(['{}. {} ({})'.format(i+1, t[0], t[1]) for i, t in enumerate(tasks)]) or 'n/a'
    lines = [
        'Topo Memory',
        'total nodes: {}'.format(mem['total_nodes']),
        'visited: {}'.format(mem['visited_waypoints']),
        'frontiers: {}'.format(mem['frontiers']),
        'objects: {}'.format(mem['objects']),
        'rooms: {}'.format(mem['rooms']),
        'landmarks: {}'.format(mem.get('landmarks', 0)),
        '',
        'Current Goal',
        'goal: {}/object'.format(goal.get('target_object', 'n/a')),
        'target node: {}'.format(st.get('target_node_id') or 'n/a'),
        '',
        'Current Action',
        'low action: {}'.format(st.get('low_action')),
        'agent action: {}'.format(st.get('agent_action')),
        '',
    ]
    lines.extend(fmt_scores('Top object CLIP', perception.get('goal_scores', [])))
    lines.extend(fmt_scores('Top room CLIP', perception.get('room_scores', [])))
    lines.extend(fmt_scores('Top landmark CLIP', perception.get('landmark_scores', [])))
    lines.extend(['', 'Episode Tasks', task_text])
    info.axis('off')
    info.text(0.02, 0.98, '\n'.join(lines), va='top', ha='left', fontsize=9, family='monospace')

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return None
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', default='data/logs/goat_topo/topo_trace_semantic.json')
    ap.add_argument('--out-dir', default='data/logs/goat_topo/viz_semantic')
    ap.add_argument('--stride', type=int, default=2)
    args = ap.parse_args()

    trace = json.load(open(args.trace))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlim, zlim = collect_limits(trace['steps'])

    final_png = out_dir / 'topo_map_final.png'
    draw_step(trace, len(trace['steps']) - 1, xlim, zlim, final_png)

    key_steps = [0, min(10, len(trace['steps'])-1), min(25, len(trace['steps'])-1), min(50, len(trace['steps'])-1), len(trace['steps'])-1]
    key_frames = []
    for i in sorted(set(key_steps)):
        p = out_dir / 'topo_map_step_{:03d}.png'.format(i)
        draw_step(trace, i, xlim, zlim, p)
        key_frames.append(str(p))

    frames = []
    stride = max(1, args.stride)
    for i in range(0, len(trace['steps']), stride):
        frames.append(draw_step(trace, i, xlim, zlim))
    if (len(trace['steps']) - 1) % stride != 0:
        frames.append(draw_step(trace, len(trace['steps']) - 1, xlim, zlim))
    gif = out_dir / 'topo_map_semantic_growth.gif'
    imageio.mimsave(gif, frames, duration=0.18)

    summary = {
        'trace': args.trace,
        'steps': len(trace['steps']),
        'final_png': str(final_png),
        'gif': str(gif),
        'key_frames': key_frames,
        'final_memory': trace.get('final_memory', {}),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
