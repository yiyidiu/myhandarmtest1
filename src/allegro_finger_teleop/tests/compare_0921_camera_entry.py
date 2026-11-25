"""Resolve both release camera command lines through argparse without starting a camera or GUI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

PROJECT = Path(__file__).resolve().parents[3]
ORIGINAL = Path(os.environ.get('FR3_SEPTEMBER_CHECKOUT', PROJECT.parent / 'whole-wrist'))
WRAPPERS = ('run_pouring_observation.sh', 'run_hand_imu_observation.sh', 'run_whole_wrist_teleop.sh')


def release_environment(tmp, env):
    # Run the unmodified release wrappers, replacing only the downstream recorder.
    replica = tmp / 'release'
    (replica / 'scripts').mkdir(parents=True)
    (replica / 'devel').symlink_to(ORIGINAL / 'devel')
    shutil.copy2(ORIGINAL / 'run.sh', replica / 'run.sh')
    for name in WRAPPERS:
        shutil.copy2(ORIGINAL / 'scripts' / name, replica / 'scripts' / name)
    snapshot = tmp / 'environment.json'
    (replica / 'scripts/run_teleop_feedback_recording.sh').write_text('''#!/usr/bin/env bash
/usr/bin/python3 - <<'END_ENV'
import os, json
from pathlib import Path
Path(os.environ['AUDIT_ENV']).write_text(json.dumps({k: v for k,v in os.environ.items() if k.startswith('TELEOP_')}))
END_ENV
''')
    subprocess.run(['bash', str(replica / 'run.sh')], check=True, stdout=subprocess.DEVNULL,
                   env=dict(env, AUDIT_ENV=str(snapshot)))
    return json.loads(snapshot.read_text())


def original_camera_command():
    # These sections set only argv/affinity. The subsequent camera call is intercepted at Python.
    source = (ORIGINAL / 'scripts/run_live_human_teleop_validation.sh').read_text()
    setup = ['''source teleop_machine.env
export MEDIAPIPE_PYTHON="$TELEOP_MEDIAPIPE_PYTHON"
export MEDIAPIPE_ENV_PREFIX="$(dirname "$(dirname "$MEDIAPIPE_PYTHON")")"
source scripts/teleop_cpu_profile.sh
hamer_command=("$AUDIT_PROBE")
asset_root="$TELEOP_HAMER_ASSET_ROOT"
checkpoint="$asset_root/hamer_ckpts/checkpoints/hamer.ckpt"
control_reference="$TELEOP_CONTROL_REFERENCE"''']
    for array in ('diagnostic_arguments=()', 'camera_affinity_arguments=()'):
        start = source.index(array)
        setup.append(source[start:source.index('\nfi', start) + 3])
    start = source.index('"${hamer_command[@]}" perception_hamer/scripts/')
    end = source.index('  2>&1 | tee', start)
    setup.append(source[start:end].rstrip().rstrip('\\'))
    return '\n'.join(setup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    upstream = subprocess.check_output(['git', '-C', str(ORIGINAL), 'rev-parse', 'HEAD'], text=True).strip()
    assert upstream == '77d6accedd5824492e32131f24dad27ee769a15b'
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'USER', 'LANG')}
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)
        probe = tmp / 'camera_args'
        python = PROJECT / '.runtime/fr3/envs/hamer_rtx2060/bin/python'
        probe.write_text('#!' + str(python) + '\n' + '''import os, runpy, sys, json
from pathlib import Path
source = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(source.parents[2]))
module = runpy.run_path(str(source), run_name='audit_camera_parser')
Path(os.environ['AUDIT_PARSED']).write_text(json.dumps(vars(module['_parse_args']()), default=str))
''')
        probe.chmod(0o700)
        env.update(AUDIT_PROBE=str(probe), TELEOP_WRIST_DIAGNOSTICS_DIRECTORY=str(tmp / 'perception'))
        old_env = dict(env, **release_environment(tmp, env))
        subprocess.run(['bash', '-ec', original_camera_command()], cwd=ORIGINAL, check=True,
                       env=dict(old_env, AUDIT_PARSED=str(tmp / 'old.json')))
        fr3 = '''source scripts/teleop_cpu_profile.sh
export FR3_VISION_CPUS="$teleop_inference_cpus" FR3_SIDECAR_CPUS="$teleop_sidecar_cpus" FR3_CAPTURE_CPUS="$teleop_capture_cpus"
exec bash src/allegro_finger_teleop/scripts/fingers_camera.sh 5031'''
        subprocess.run(['bash', '-ec', fr3], cwd=PROJECT, check=True, env=dict(env, AUDIT_PARSED=str(tmp / 'new.json'),
                       TELEOP_HAMER_PYTHON=str(probe), ALLEGRO_TELEOP_PROJECT=str(PROJECT)))
        old, new = [json.loads((tmp / name).read_text()) for name in ('old.json', 'new.json')]
    def digest(path):
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha.update(chunk)
        return sha.digest()

    for asset in ('hamer_ckpts/checkpoints/hamer.ckpt', 'hamer_ckpts/model_config.yaml',
                  'data/mano_mean_params.npz', 'data/mano/MANO_RIGHT.pkl'):
        first, second = (Path(params['data_root']) / asset for params in (old, new))
        assert digest(first) == digest(second), asset
    # Resolve absolute helper paths to the installed packages on each machine.
    for key in ('display_helper_python', 'realsense_sdk_site_packages'):
        paths = [Path(params[key]) for params in (old, new)]
        assert all(p.exists() for p in paths), paths
        prefix = [p.parent.parent if key == 'display_helper_python' else p.parents[2] for p in paths]
        assert paths[0].relative_to(prefix[0]) == paths[1].relative_to(prefix[1]), paths
        for params in (old, new):
            params[key] = str(paths[0].relative_to(prefix[0]))
    for parsed in (old, new):
        parsed['data_root'] = parsed['checkpoint'] = 'verified same model files'
    differences = {key: [old.get(key), new.get(key)] for key in sorted(old.keys() | new.keys())
                   if old.get(key) != new.get(key)}
    assert differences == {'finger_landmarks': [None, True], 'pure_view_correction': [None, False], 'palm_hull_depth': [None, True],
                           'orientation_filter_large_angle_mode': ['reject', 'confirm'],
                           'teleop_udp_port': [5010, 5031]}, differences
    result = {'common_equal': sum(old[k] == new[k] for k in old.keys() & new.keys()), 'differences': differences}
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
