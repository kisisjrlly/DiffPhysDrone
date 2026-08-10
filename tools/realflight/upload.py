#!/usr/bin/env python3
"""Upload a local file to the onboard computer over SFTP."""

import os
import sys

import paramiko


HOST = os.environ.get("RS_HOST", "192.168.1.208")
USER = os.environ.get("RS_USER", "xgg")
PASS = os.environ.get("RS_SSH_PASS", "")


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: upload.py <local> <remote>", file=sys.stderr)
        sys.exit(2)
    if not PASS:
        print("RS_SSH_PASS not set", file=sys.stderr)
        sys.exit(2)
    local, remote = sys.argv[1], sys.argv[2]
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
    sftp = client.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    client.close()
    print(f"uploaded {local} -> {remote}")


if __name__ == "__main__":
    main()
