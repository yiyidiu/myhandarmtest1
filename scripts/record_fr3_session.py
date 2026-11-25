#!/usr/bin/env python3
"""Record the desktop, camera streams and ROS control path in one session."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def stop(process):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 35), (signal.SIGTERM, 10), (signal.SIGKILL, 3)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def read_summary(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def inspect_recording(directory, result):
    import rosbag

    messages = {}
    try:
        with rosbag.Bag(str(directory / 'robot.bag')) as bag:
            for topic, info in bag.get_type_and_topic_info().topics.items():
                messages[topic] = info.message_count
    except (OSError, rosbag.ROSBagException) as exc:
        result['bag_error'] = str(exc)
    result['messages'] = messages
    video = read_summary(directory / 'screen_summary.json')
    clock = read_summary(directory / 'clock_and_hand_summary.json')
    result['video'] = video
    result['time_alignment'] = clock
    rgbd = directory / 'rgbd.db3'
    result['rgbd_bytes'] = rgbd.stat().st_size if rgbd.is_file() else 0
    result['perception_files'] = sorted(str(p.relative_to(directory)) for p in
                                        (directory / 'perception').rglob('frames.jsonl') if p.stat().st_size)
    checks = {
        'joint_states': messages.get('/fr3_allegro/joint_states', 0),
        'camera_packets': messages.get('/fr3_allegro/camera_packet', 0),
        'screen': video.get('eos') and not video.get('error') and
                  video.get('container_timeline', {}).get('usable_frame_mapping'),
        'simulation_clock': clock.get('sim_clock_messages', 0),
        'camera_timeline': clock.get('camera_rows', 0),
    }
    if result['mode'] == 'live':
        streams = set()
        if rgbd.is_file():
            try:
                with sqlite3.connect(rgbd.as_uri() + '?mode=ro', uri=True) as db:
                    streams = {row[0] for row in db.execute(
                        'SELECT t.name FROM topics t WHERE t.name LIKE ? '
                        'AND EXISTS (SELECT 1 FROM messages m WHERE m.topic_id=t.id)', ('%/image/data',))}
            except sqlite3.Error as exc:
                result['rgbd_error'] = str(exc)
        result['rgbd_image_topics'] = sorted(streams)
        checks.update(rgb=any('/Color_' in s for s in streams), depth=any('/Depth_' in s for s in streams),
                      perception=result['perception_files'])
    result['missing'] = [name for name, ok in checks.items() if not ok]
    result['recording_complete'] = not result['missing']


def revision():
    if (ROOT / '.git').exists():
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        changed = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
        return commit, changed
    archived = (ROOT / 'reproduction/fr3/source-revision.txt').read_text().strip()
    return (archived if len(archived) == 40 else 'fr3-allegro-teleop-v2026.11.25'), None


def main():
    mode, *launch = sys.argv[1:]
    parent = Path(os.environ.get('FR3_RECORDING_ROOT', Path.home() / 'handarm/results/fr3-teleop-recordings')).expanduser().resolve()
    parent.mkdir(exist_ok=True, parents=True)
    directory = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S-'), dir=parent))
    print('本次录制目录', directory, flush=True)
    env = os.environ.copy()
    env['FR3_RECORDING_DIRECTORY'] = str(directory)
    env['TELEOP_ROBOT_FEEDBACK_FILE'] = str(directory / 'robot_feedback.json')
    if mode == 'live':
        env['TELEOP_RGBD_RECORDING_FILE'] = str(directory / 'rgbd.db3')
        env['TELEOP_WRIST_DIAGNOSTICS_DIRECTORY'] = str(directory / 'perception')
    affinity = ['taskset', '-c', env['FR3_VISION_CPUS']] if env.get('FR3_VISION_CPUS') else []
    helpers = ROOT / 'scripts/feedback_capture'
    screen = robot = clock = None
    code, changed = revision()
    result = {
        'mode': mode, 'wrist': env.get('FR3_WRIST_OBSERVATION', 'baseline'),
        'control_point': env.get('FR3_CONTROL_POINT', 'wrist'), 'scene': Path(env['FR3_WORLD']).name,
        'code': code, 'code_had_uncommitted_changes': changed,
        'started_unix_s': time.time(),
    }

    def interrupted(sig, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (directory / 'screen.log').open('w') as log:
            screen = subprocess.Popen(affinity + [sys.executable, str(helpers / 'record_screen_clock.py'), str(directory)],
                                      env=env, start_new_session=True, stdout=log, stderr=log)
        deadline = time.monotonic() + 12
        while not (directory / 'screen_ready.json').exists():
            if screen.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('录屏启动失败，详见 ' + str(directory / 'screen.log'))
            time.sleep(.1)
        robot = subprocess.Popen(launch, env=env, start_new_session=True)
        with (directory / 'clock.log').open('w') as log:
            clock = subprocess.Popen(affinity + [sys.executable, str(helpers / 'record_ros_clock_and_hand.py'), str(directory)],
                                     env=env, start_new_session=True, stdout=log, stderr=log)
        while robot.poll() is None:
            if clock.poll() is not None:
                raise RuntimeError('时钟记录提前结束，详见 ' + str(directory / 'clock.log'))
            if screen.poll() is not None:
                raise RuntimeError('录屏提前结束，详见 ' + str(directory / 'screen.log'))
            time.sleep(.15)
        if robot.returncode:
            raise RuntimeError('ROS 启动程序退出，状态码 ' + str(robot.returncode))
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop(robot)
        stop(clock)
        stop(screen)
        result['ended_unix_s'] = time.time()
        inspect_recording(directory, result)
        (directory / 'session.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print('录制已保存。' if result['recording_complete'] else '本次录制有缺项，详见 session.json。', directory, flush=True)
    return 0 if result['recording_complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
