from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Proxy Structivize rendering through the STRUDEL CLI.")
    parser.add_argument(
        "--render-script",
        default=os.environ.get("STRUCTIVIZE_RENDER_SCRIPT"),
        help="Path to structivize render_batch.py. Can also be set via STRUCTIVIZE_RENDER_SCRIPT.",
    )
    args, passthrough = parser.parse_known_args()

    if not args.render_script:
        print("Missing render script path. Use --render-script or set STRUCTIVIZE_RENDER_SCRIPT.", file=sys.stderr)
        return 1

    result = subprocess.run([sys.executable, args.render_script, *passthrough])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
