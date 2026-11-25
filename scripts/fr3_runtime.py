#!/usr/bin/env python3
"""Restore the two frozen, relocatable vision environments and the FR3 baseline assets."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import restore_release_runtime as release

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".runtime" / "fr3"
MODELS = ROOT / "perception_hamer" / "_DATA"
MANO = "data/mano/MANO_RIGHT.pkl"
MANIFEST = release.MANIFEST


def fingerprints(root):
    return {name: [p.stat().st_size, p.stat().st_mtime_ns] for name in MANIFEST["model_files"]
            for p in (root / name,) if p.is_file()}


def restore_models(kit, cache):
    marker = STATE / "models.json"
    checks = {k: v["sha256"] for k, v in MANIFEST["model_files"].items()}
    current = fingerprints(MODELS)
    if marker.is_file() and len(current) == len(checks):
        if json.loads(marker.read_text()) == [checks, current]:
            return
    candidates = [MODELS, kit / "_DATA"]
    if os.environ.get("FR3_ASSETS_FROM"):
        candidates.insert(0, Path(os.environ["FR3_ASSETS_FROM"]).expanduser())
    full = next((p for p in candidates if all((p / name).is_file() for name in checks)), None)
    licensed = [Path(os.environ["FR3_MANO_FILE"]).expanduser()] if os.environ.get("FR3_MANO_FILE") else []
    licensed += [p / MANO for p in candidates] + [
        ROOT / "MANO_RIGHT.pkl", ROOT.parent / "MANO_RIGHT.pkl",
        Path.home() / "Downloads/MANO_RIGHT.pkl", Path.home() / "下载/MANO_RIGHT.pkl"]
    mano = next((p for p in licensed if release.checked(p, MANIFEST["model_files"][MANO])), None)
    if mano is None:
        raise RuntimeError("实时模式需要本基线授权的 MANO_RIGHT.pkl。把文件放到 run.sh 同一目录再运行即可。"
                           "仅测臂手可直接 ./run.sh --replay 。")
    print("正在核对 HaMeR 权重与授权的 MANO 模型。", flush=True)
    release.restore_models(SimpleNamespace(assets_from=full, mano=mano), cache)
    store(marker, [checks, fingerprints(MODELS)])


def store(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


def import_owner_archive(archive, cache):
    # Extract only the release manifest's regular files, to fixed destinations.
    # This avoids a separate 9 GB unpacked copy and leaves the travel archive intact.
    files = {"fr3-owner/_DATA/" + name: (MODELS / name, spec)
             for name, spec in MANIFEST["model_files"].items()}
    for environment in MANIFEST["environments"].values():
        for part in environment["parts"]:
            name = part["name"]
            files["fr3-owner/cache/" + name] = (cache / name, part)
    wheel = MANIFEST["smplx"]
    files["fr3-owner/cache/" + wheel["filename"]] = (cache / wheel["filename"], wheel)
    marker = STATE / "owner-archive.json"
    signature = [str(archive), archive.stat().st_size, archive.stat().st_mtime_ns]
    if marker.is_file() and json.loads(marker.read_text()) == signature and all(p.is_file() for p, _ in files.values()):
        return
    print("正在从个人迁移包恢复本版视觉依赖。", flush=True)
    with tarfile.open(archive, "r:") as stream:
        for name, (dest, spec) in files.items():
            if release.checked(dest, spec):
                continue
            member = stream.getmember(name)
            if not member.isfile() or member.size != spec.get("size", member.size):
                raise RuntimeError("个人迁移包中有文件与基线不符：" + name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
                temporary = Path(out.name)
                try:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        out.write(chunk)
                    if digest.hexdigest() != spec["sha256"]:
                        raise RuntimeError("个人迁移包校验失败：" + name)
                    out.flush()
                    temporary.replace(dest)
                finally:
                    temporary.unlink(missing_ok=True)
    store(marker, signature)


def verify_env(name, python):
    code = ("import torch,cv2,smplx,pyrealsense2; assert torch.__version__ == '2.2.0+cu118'"
            if name == "hamer_rtx2060" else
            "import cv2,mediapipe as mp; assert mp.__version__ == '0.10.21'; "
            "assert cv2.__version__ == '4.11.0'; mp.solutions.hands.Hands().close()")
    subprocess.run([str(python), "-c", code], check=True)


def select_env(alias, destination):
    tmp = alias.with_suffix(".next")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(os.path.relpath(destination, alias.parent))
    tmp.replace(alias)


def restore_env(name, cache):
    base = STATE / "envs"
    base.mkdir(exist_ok=True)
    alias = base / name
    external_marker = STATE / (name + "-external.json")
    if os.environ.get("FR3_EXISTING_ENV_ROOT"):
        external = (Path(os.environ["FR3_EXISTING_ENV_ROOT"]).expanduser() / name).resolve()
        verify_env(name, external / "bin/python")
        select_env(alias, external)
        store(external_marker, str(external))
        return
    if external_marker.is_file() and json.loads(external_marker.read_text()) == str(alias.resolve()) and (alias / "bin/python").is_file():
        return
    marker = alias / ".fr3-prefix"
    if marker.is_file() and marker.read_text() == str(alias.resolve()) and (alias / "bin/python").is_file():
        return
    # conda-unpack embeds the absolute prefix. Keep the restored directory at its
    # final path and atomically select it only after both installation and imports pass.
    dest = Path(tempfile.mkdtemp(dir=base, prefix=name + "-"))
    try:
        python = release.restore_environment(name, dest, cache)
        verify_env(name, python)
        (dest / ".fr3-prefix").write_text(str(dest.resolve()))
        select_env(alias, dest)
    except BaseException:
        shutil.rmtree(dest)
        raise


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    candidates = [ROOT / "fr3-owner", ROOT.parent / "fr3-owner", Path.home() / "fr3-owner"]
    kit = next((p for p in candidates if p.is_dir()), candidates[0])
    # The checkout is always sufficient for public downloads. A private owner's
    # travel kit optionally seeds the verified archives without a network transfer.
    cache = Path(os.environ.get("FR3_CACHE_DIR", STATE / "downloads")).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (STATE / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if kit.is_dir():
            parts = [p["name"] for env in MANIFEST["environments"].values() for p in env["parts"]]
            for name in parts + [MANIFEST["smplx"]["filename"]]:
                source, dest = kit / "cache" / name, cache / name
                if source.is_file() and not dest.exists() and not dest.is_symlink():
                    dest.symlink_to(source.resolve())
        else:
            archives = [p.with_suffix(".tar") for p in candidates]
            archives.extend([Path.home() / "Downloads/fr3-owner.tar", Path.home() / "下载/fr3-owner.tar"])
            archive = next((p for p in archives if p.is_file()), None)
            if archive:
                import_owner_archive(archive, cache)
        restore_models(kit, cache)
        for name in MANIFEST["environments"]:
            restore_env(name, cache)


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, RuntimeError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
