#!/usr/bin/python3
"""Bring this session's preview in front once if the Gazebo startup covers it."""
import os
from pathlib import Path
import re
import subprocess
import time


def window_ids(tree):
    own_master = ("ROS_MASTER_URI=" + os.environ["ROS_MASTER_URI"].rstrip("/")).encode()
    windows = {}
    for xid, title in re.findall(r'\b(0x[\da-fA-F]+) "([^"]+)"', tree):
        if title == "Gazebo":
            kind = "sim"
        elif title.startswith(("D455 HaMeR", "HaMeR Q_t B0")):
            kind = "camera"
        else:
            continue
        pid = subprocess.run(["xprop", "-id", xid, "_NET_WM_PID"], capture_output=True,
                             text=True).stdout.rpartition(" = ")[2].strip()
        if not pid.isdigit():
            continue
        try:
            environment = Path("/proc", pid, "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(e.rstrip(b"/") == own_master for e in environment):
            windows[kind] = int(xid, 16)
    return windows


def is_mapped(xid):
    result = subprocess.run(["xwininfo", "-id", hex(xid)], capture_output=True,
                            text=True, env=dict(os.environ, LC_ALL="C"))
    return "Map State: IsViewable" in result.stdout


def main():
    import gi
    gi.require_version("Gdk", "3.0")
    gi.require_version("GdkX11", "3.0")
    gi.require_version("Wnck", "3.0")
    from gi.repository import Gdk, GdkX11, GLib, Wnck

    display = Gdk.Display.get_default()
    if not isinstance(display, GdkX11.X11Display):
        return
    screen = Wnck.Screen.get_default()
    deadline = time.monotonic() + 60.0
    both_visible_at = None
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True)
        ids = window_ids(tree.stdout)
        if len(ids) == 2 and all(is_mapped(xid) for xid in ids.values()):
            if both_visible_at is None:
                both_visible_at = time.monotonic()
            screen.force_update()
            if screen.get_window_manager_name():
                camera = Wnck.Window.get(ids["camera"])
                active = screen.get_active_window()
                if camera is not None and Wnck.Window.get(ids["sim"]) is not None:
                    if active is not None and active.get_xid() == ids["sim"]:
                        camera.activate(GdkX11.x11_get_server_time(Gdk.get_default_root_window()))
                        display.sync()
                        return
                    if time.monotonic() - both_visible_at >= 1.5:
                        return
            else:
                # An unmanaged X11 display has no focus policy or window frames.
                camera = GdkX11.X11Window.foreign_new_for_display(display, ids["camera"])
                if camera is not None:
                    camera.raise_()
                    display.sync()
                return
        time.sleep(.25)


if __name__ == "__main__":
    main()
