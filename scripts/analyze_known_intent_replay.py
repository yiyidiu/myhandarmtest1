"""Compare measured Gazebo motion on the ORIGINAL uninterrupted time axis."""
from pathlib import Path
import gzip
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

P = Path(__file__).resolve().parents[1] / '.runtime/known_intent_replay_20260913'
ROOT = P.parents[1]
labels = json.loads((ROOT / '.runtime/hand_motion_observation_20260912T145617_iJYZ/analysis_feedback/intent_data_definition.json').read_text())['annotations']['test']
datasets = {}
for name in ['baseline', 'oracle_continuous']:
    compressed = P / (name + '_actual.json.gz')
    if compressed.exists():
        with gzip.open(compressed, 'rt') as stream:
            report = json.load(stream)
    else:
        report = json.loads((P / (name + '_actual.json')).read_text())
    rows = [r for r in report['rows'] if r.get('trend', {}).get('current_quaternion_xyzw')]
    t = np.array([r['t'] for r in rows])
    zero = R.from_quat(rows[0]['trend']['robot_zero_quaternion_xyzw'])
    actual = R.from_quat([r['trend']['current_quaternion_xyzw'] for r in rows])
    target = R.from_quat([r['trend']['target_quaternion_xyzw'] for r in rows])
    def components(rot):
        xyz = (zero.inv() * rot).as_euler('XYZ')
        return np.rad2deg(np.column_stack([-xyz[:, 1], -xyz[:, 2], xyz[:, 0]]))
    datasets[name] = dict(t=t, rows=rows, actual=actual, target=target,
                          angles=components(actual), target_angles=components(target),
                          actual_c=np.rad2deg((zero.inv() * actual).magnitude()),
                          target_c=np.rad2deg((zero.inv() * target).magnitude()), zero=zero,
                          actual_position=np.array([r['trend']['current_position'] for r in rows]),
                          target_position=np.array([r['trend']['target_position'] for r in rows]),
                          report=report)
summary = {'scope': 'Actual Gazebo end-effector measurements, same fixed-C coordinate convention. Sparse human labels, not an automatic-intent result; original timeline and continuous history. Translation and fingers identical between conditions.',
           'rows': [], 'runs': {}}
for name, d in datasets.items():
    summary['runs'][name] = dict(fault=d['report']['fault'], last_time=d['report']['final_input_time'],
        scheduling_lateness_p95_ms=d['report']['scheduling_lateness_p95_ms'],
        zero_position=d['rows'][0]['trend']['robot_zero_position'],
        zero_quaternion=d['zero'].as_quat().tolist(), observations=len(d['t']))
for a in labels:
    result = dict(a)
    for name, d in datasets.items():
        indices = np.flatnonzero((d['t'] >= a['start']) & (d['t'] <= a['end']))
        if not len(indices):
            continue
        first = indices[0]
        actual_change = (d['actual'][first].inv() * d['actual'][indices]).magnitude()
        target_change = (d['target'][first].inv() * d['target'][indices]).magnitude()
        error = (d['target'][indices].inv() * d['actual'][indices]).magnitude()
        p_error = np.linalg.norm(d['actual_position'][indices] - d['target_position'][indices], axis=1)
        main = {'pitch': 0, 'roll': 1, 'yaw': 2}.get(a['label'])
        other = [i for i in range(3) if i != main]
        angle_change = d['angles'][indices] - d['angles'][first]
        counts = {}
        for i in indices:
            reason = d['rows'][i].get('geometry', {}).get('reason', 'missing')
            counts[reason] = counts.get(reason, 0) + 1
        result[name] = dict(
            actual_start_c_components_deg=d['angles'][first].tolist(),
            actual_end_c_components_deg=d['angles'][indices[-1]].tolist(),
            actual_start_c_total_deg=float(d['actual_c'][first]),
            actual_excursion_deg=float(np.rad2deg(actual_change.max())),
            target_excursion_deg=float(np.rad2deg(target_change.max())),
            actual_other_component_change_deg=float(np.linalg.norm(angle_change[:, other], axis=1).max()),
            target_tracking_error_p95_deg=float(np.rad2deg(np.percentile(error, 95))),
            position_error_p95_mm=float(1000 * np.percentile(p_error, 95)),
            actual_position_travel_mm=float(1000 * np.linalg.norm(np.diff(d['actual_position'][indices],axis=0),axis=1).sum()),
            main_change_deg=None if main is None else float(np.abs(angle_change[:, main]).max()),
            main_range_deg=None if main is None else float(np.ptp(d['angles'][indices, main])),
            target_main_range_deg=None if main is None else float(np.ptp(d['target_angles'][indices, main])),
            reason_counts=counts)
    summary['rows'].append(result)
(P / 'actual_comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

font_file = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_file)
font_name = font_manager.FontProperties(fname=font_file).get_name()
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [font_name],
                     'axes.unicode_minus': False, 'font.size': 10})
fig, axes = plt.subplots(4, 2, figsize=(12, 13))
panels = [
    (70.72, 72.4, 'C后静止：相对原C姿态', 'c', None),
    (83., 86.5, '向下平移：末端额外转动', 'change', None),
    (103., 107.8, '向右平移：末端额外转动', 'change', None),
    (153., 163.5, '主动俯仰：其他角分量变化', 'other', 0),
    (169., 172., '较好俯仰：主俯仰分量', 'main', 0),
    (199., 202., '主动横滚：其他角分量变化', 'other', 1),
    (199., 202., '主动横滚：主横滚分量', 'main', 1),
    (273., 278., '返回标定附近：相对原C姿态', 'c', None),
]
colors = {'baseline': '#80858b', 'oracle_continuous': '#007c91'}
for ax, (start, end, title, mode, main) in zip(axes.flat, panels):
    for name, d in datasets.items():
        ids = np.flatnonzero((d['t'] >= start) & (d['t'] <= end))
        if not len(ids): continue
        first = ids[0]
        for kind, style in [('actual','-'),('target','--')]:
            angles = d['angles'] if kind == 'actual' else d['target_angles']
            if mode == 'c':
                y = d[kind + '_c'][ids]
            elif mode == 'change':
                y = np.rad2deg((d[kind][first].inv() * d[kind][ids]).magnitude())
            elif mode == 'main':
                y = angles[ids, main]
            else:
                other = [j for j in range(3) if j != main]
                y = np.linalg.norm((angles[ids] - angles[first])[:,other],axis=1)
            ax.plot(d['t'][ids],y,style,color=colors[name],lw=1.8 if kind=='actual' else 1,alpha=1 if kind=='actual' else .6)
    ax.set_title(title, loc='left')
    covered = [(max(start, a['start']), min(end, a['end'])) for a in labels
               if a['end'] >= start and a['start'] <= end]
    cursor = start
    for begin, finish in sorted(covered):
        if begin > cursor:
            ax.axvspan(cursor, begin, color='#cccccc', alpha=.2, zorder=-2)
        cursor = max(cursor, finish)
    if cursor < end:
        ax.axvspan(cursor, end, color='#cccccc', alpha=.2, zorder=-2)
    ax.set_ylabel('角度（°）')
    ax.grid(alpha=.17)
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)//60}:{x%60:04.1f}'))
fig.suptitle('同一连续回放：人工动作标注能否改善实际末端运动',fontsize=16,y=.985)
fig.text(.5,.958,'灰色：原输入　青色：人工标注辅助　实线：实际末端　虚线：控制目标',ha='center',fontsize=11)
fig.text(.5,.022,'横轴为原录像时间，未移动曲线；C只在原标定时建立。\n浅灰背景：未标注时段，恢复原输入。这是诊断实验，不代表自动意图模型已经可用。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.05,1,.945),h_pad=2.1)
fig.savefig(P / 'actual_motion_comparison.png',dpi=160)
fig.savefig(P / 'actual_motion_comparison.pdf')
for row in summary['rows']:
    b=row.get('baseline');c=row.get('oracle_continuous')
    if not b or not c:continue
    print(row['id'], row['label'], 'extra',round(b['actual_other_component_change_deg'],2),'->',round(c['actual_other_component_change_deg'],2),
          'main',b['main_range_deg'],c['main_range_deg'],'position error',round(b['position_error_p95_mm'],1),round(c['position_error_p95_mm'],1))
print('runs', summary['runs'])
