#!/usr/bin/env python3
"""macOS listener-aware entry point for the Qwen MTP smoke ladder.

A cleanly terminated TCP server can leave recently used sockets in TIME_WAIT.
That state can temporarily reject a plain bind even though no process is
listening. The smoke gate cares about stale listeners and orphan processes,
not harmless kernel bookkeeping, so this entry point replaces the bind-based
probe with an active listener probe before running the shared ladder.

``QWEN_MTP_SMOKE_ARMS`` may select a comma-separated subset for a focused
promotion run after the complete four-arm startup ladder has passed.
"""

from __future__ import annotations

import os
import socket
import sys
import time

import qwen_mtp_combo_smoke as smoke


def port_has_listener(port: int) -> bool:
    """Return True only while something accepts TCP connections on ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_listener_release(port: int, timeout_s: float = 20.0) -> bool:
    """Wait until the server listener disappears, ignoring TCP TIME_WAIT."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not port_has_listener(port):
            return True
        time.sleep(0.25)
    return not port_has_listener(port)


def select_arms_from_environment() -> None:
    requested = os.getenv("QWEN_MTP_SMOKE_ARMS", "").strip()
    if not requested:
        return
    names = [name.strip() for name in requested.split(",") if name.strip()]
    available = {arm.name: arm for arm in smoke.ARMS}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"unknown Qwen MTP smoke arm(s): {', '.join(unknown)}")
    if not names:
        raise SystemExit("QWEN_MTP_SMOKE_ARMS selected no arms")
    smoke.ARMS = tuple(available[name] for name in names)


smoke.wait_for_port_release = wait_for_listener_release
select_arms_from_environment()

if __name__ == "__main__":
    sys.exit(smoke.main())
