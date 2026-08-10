#!/usr/bin/env python3
"""Run one-shot commands on the onboard computer over SSH.

Usage:
    RS_SSH_PASS=... RS_HOST=192.168.1.208 python3 remote.py 'cmd1' 'cmd2'

Interactive PTY over SSH loses keystrokes on this setup, so commands are
executed as a single non-interactive session with a PTY for ROS output.
"""

import os
import sys

import paramiko


HOST = os.environ.get("RS_HOST", "192.168.1.208")
USER = os.environ.get("RS_USER", "xgg")
PASS = os.environ.get("RS_SSH_PASS", "")


def run(cmd: str, timeout: int = 40) -> tuple[int, str, str]:
    if not PASS:
        print("RS_SSH_PASS not set", file=sys.stderr)
        sys.exit(2)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username=USER,
        password=PASS,
        timeout=12,
        banner_timeout=12,
        auth_timeout=12,
    )
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    client.close()
    return code, out, err


def main() -> None:
    cmds = sys.argv[1:] or ["echo hello"]
    for cmd in cmds:
        print(f"### CMD: {cmd}")
        code, out, err = run(cmd)
        print(f"### EXIT={code}")
        if out.strip():
            print(out.rstrip())
        if err.strip():
            print("### STDERR:\n" + err.rstrip())


if __name__ == "__main__":
    main()
