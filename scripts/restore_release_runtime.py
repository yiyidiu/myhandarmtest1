#!/usr/bin/env python3
"""Restore pinned public runtime dependencies and import the owner's MANO file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'reproduction/20260917/runtime_reference.json').read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked(path, expected):
    return (path.is_file() and ('size' not in expected or path.stat().st_size == expected['size'])
            and sha256(path) == expected['sha256'])


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if expected and checked(path, expected):
        return path
    if expected and path.exists() and path.stat().st_size >= expected.get('size', float('inf')):
        raise RuntimeError('Existing download differs from this release; move it aside: ' + str(path))
    print('Download: ' + path.name, flush=True)
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '20',
                    '--continue-at', '-', '--output', str(path), url], check=True)
    if expected and not checked(path, expected):
        raise RuntimeError('Release dependency checksum mismatch: ' + str(path))
    return path


def restore_environment(name, destination, cache):
    marker = destination / '.teleop-restored.json'
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved.get('environment_release') == MANIFEST['environment_release_tag']:
            return destination / 'bin/python'
        raise RuntimeError('Environment belongs to another release: ' + str(destination))
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Will not overwrite a nonempty environment: ' + str(destination))
    base = ('https://github.com/' + MANIFEST['github_repository'] + '/releases/download/'
            + MANIFEST['environment_release_tag'] + '/')
    parts = [download(base + part['name'], cache / part['name'], part)
             for part in MANIFEST['environments'][name]['parts']]
    destination.mkdir(parents=True, exist_ok=True)
    print('Restore environment: ' + str(destination), flush=True)
    cat = subprocess.Popen(['cat', *map(str, parts)], stdout=subprocess.PIPE)
    try:
        subprocess.run(['tar', '-xz', '-C', str(destination)], stdin=cat.stdout, check=True)
    finally:
        cat.stdout.close()
    if cat.wait() != 0:
        raise RuntimeError('Could not read all environment volumes')
    python = destination / 'bin/python'
    subprocess.run([str(python), str(destination / 'bin/conda-unpack')], check=True)
    wheel = MANIFEST['smplx']
    wheel_path = download(wheel['url'], cache / wheel['filename'], wheel)
    # The original licensor's distribution is used; it is not republished here.
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', str(wheel_path)], check=True)
    marker.write_text(json.dumps({'environment_release': MANIFEST['environment_release_tag']}) + '\n')
    return python


def restore_models(args, cache):
    target = ROOT / 'perception_hamer/_DATA'
    model_files = MANIFEST['model_files']
    mano_key = 'data/mano/MANO_RIGHT.pkl'
    source = args.assets_from.resolve() if args.assets_from else None
    mano = args.mano.resolve() if args.mano else (source / mano_key if source else target / mano_key)
    if not checked(mano, model_files[mano_key]):
        raise RuntimeError('Provide your authorized baseline MANO_RIGHT.pkl with --mano FILE, '
                           'or your complete _DATA directory with --assets-from DIR. '
                           'MANO is not publicly redistributed: https://mano.is.tue.mpg.de/license.html')
    target.mkdir(parents=True, exist_ok=True)
    for relative, reference in model_files.items():
        destination = target / relative
        if checked(destination, reference):
            continue
        candidate = mano if relative == mano_key else (source / relative if source else None)
        if candidate is not None:
            if not checked(candidate, reference):
                raise RuntimeError('Supplied model differs from this baseline: ' + str(candidate))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
    public_files = [name for name in model_files if name != mano_key]
    missing = [name for name in public_files if not checked(target / name, model_files[name])]
    if missing:
        archive = download(MANIFEST['official_model_archive_url'], cache / 'hamer_demo_data.tar.gz')
        # Read only required regular files, with fixed destination paths.
        # Other upstream detector models are not needed by this live pipeline.
        with tarfile.open(archive, 'r|gz') as stream:
            for member in stream:
                name = member.name.removeprefix('./') if hasattr(str, 'removeprefix') else member.name.lstrip('./')
                relative = name[len('_DATA/'):] if name.startswith('_DATA/') else name
                if relative in missing and member.isfile():
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as src, destination.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
    for relative, reference in model_files.items():
        if not checked(target / relative, reference):
            raise RuntimeError('Required baseline model is missing or different: ' + relative)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-from', type=Path, help='Existing authorized _DATA directory')
    parser.add_argument('--mano', type=Path, help='Authorized MANO_RIGHT.pkl; other models download from the author')
    parser.add_argument('--existing-env-root', type=Path,
                        help='Reuse ENV_ROOT/hamer_rtx2060 and ENV_ROOT/mediapipe_env instead of downloading')
    parser.add_argument('--cache', type=Path, default=ROOT / '.runtime/dependency_downloads')
    args = parser.parse_args()
    cache = args.cache.resolve()
    assets = restore_models(args, cache)
    environments = {}
    for name in ('hamer_rtx2060', 'mediapipe_env'):
        python = ((args.existing_env_root.resolve() / name / 'bin/python')
                  if args.existing_env_root else restore_environment(name, ROOT / '.venvs' / name, cache))
        if not python.is_file():
            raise RuntimeError('Missing Python interpreter: ' + str(python))
        modules = ('import torch,smplx,pyrealsense2; assert torch.__version__ == "2.2.0+cu118"'
                   if name == 'hamer_rtx2060' else
                   'import mediapipe,cv2; assert mediapipe.__version__ == "0.10.21"; '
                   'assert cv2.__version__ == "4.11.0"')
        subprocess.run([str(python), '-c', modules], check=True)
        environments[name] = str(python)
    settings = {'TELEOP_HAMER_PYTHON': environments['hamer_rtx2060'],
                'TELEOP_MEDIAPIPE_PYTHON': environments['mediapipe_env'],
                'TELEOP_HAMER_ASSET_ROOT': str(assets)}
    config = ROOT / 'teleop_machine.env'
    if config.exists():
        shutil.copy2(config, config.with_suffix('.env.previous'))
    config.write_text('# Generated local paths; never commit this file.\n' +
                      ''.join('export '+key+'='+shlex.quote(value)+'\n' for key,value in settings.items()))
    print('Runtime restored. Local configuration: ' + str(config), flush=True)


if __name__ == '__main__':
    main()
