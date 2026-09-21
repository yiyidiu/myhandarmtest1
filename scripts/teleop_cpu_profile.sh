#!/usr/bin/env bash
# Shared by live control and recording. Keep physics off the vision/recording
# cores; Linux logical CPU numbering is not the same on every workstation.
teleop_control_cpus=''
teleop_physics_cpus=''
teleop_ros_cpus=''
teleop_inference_cpus=''
teleop_capture_cpus=''
teleop_sidecar_cpus=''
if command -v taskset >/dev/null 2>&1; then
  mapfile -t teleop_cpu_sets < <(/usr/bin/python3 - <<'PY'
import os
from pathlib import Path
groups = {}
try:
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        key = tuple((root / name).read_text().strip()
                    for name in ('physical_package_id', 'core_id'))
        groups.setdefault(key, []).append(cpu)
    cores = sorted(groups.values(), key=lambda values: values[0])
    if len(cores) >= 6:
        for selected in (cores[:1], cores[1:2], cores[2:-2],
                         cores[-2:], cores[2:3], cores[3:4]):
            print(','.join(str(cpu) for core in selected for cpu in core))
except OSError:
    pass  # An unavailable topology leaves scheduling to the operating system.
PY
  )
  if (( ${#teleop_cpu_sets[@]} == 6 )); then
    teleop_control_cpus="${teleop_cpu_sets[0]}"
    teleop_physics_cpus="${teleop_cpu_sets[1]}"
    teleop_ros_cpus="${teleop_cpu_sets[2]}"
    teleop_inference_cpus="${teleop_cpu_sets[3]}"
    teleop_capture_cpus="${teleop_cpu_sets[4]}"
    teleop_sidecar_cpus="${teleop_cpu_sets[5]}"
  fi
  unset teleop_cpu_sets
fi

teleop_pin_gazebo_physics() {
  [[ -n "$teleop_physics_cpus" ]] || return 0
  /usr/bin/python3 - "$teleop_physics_cpus" "${ROS_MASTER_URI:-http://localhost:11311}" <<'PY'
import socket
import subprocess
import sys
import time
import xmlrpc.client
from pathlib import Path

socket.setdefaulttimeout(2.0)
master = xmlrpc.client.ServerProxy(sys.argv[2])
deadline = time.monotonic() + 10.0
while time.monotonic() < deadline:
    try:
        code, _, uri = master.lookupNode('/teleop_cpu_profile', '/gazebo')
        if code == 1:
            code, _, pid = xmlrpc.client.ServerProxy(uri).getPid('/teleop_cpu_profile')
            if code == 1 and Path('/proc/%d/exe' % pid).resolve().name.startswith('gzserver'):
                subprocess.run(['taskset', '-apc', sys.argv[1], str(pid)],
                               check=True, stdout=subprocess.DEVNULL)
                break
    except (OSError, xmlrpc.client.Error):
        pass
    time.sleep(.2)
else:
    raise RuntimeError('Cannot reserve the Gazebo physics CPUs; stop before opening the camera')
PY
}
