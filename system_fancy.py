"""Launch the local GridWorld Studio: python system_fancy.py [--port 8502]."""

import argparse
from pathlib import Path
import subprocess
import sys


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    root = Path(__file__).resolve().parent
    print(f"GridWorld Studio: http://localhost:{args.port}", flush=True)
    try:
        return subprocess.call([
            sys.executable, "-m", "streamlit", "run", str(root / "web" / "fancy_app.py"),
            "--server.address", "127.0.0.1", "--server.port", str(args.port),
            "--server.headless", str(args.no_browser).lower(),
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light", "--theme.primaryColor", "#24705a",
            "--theme.backgroundColor", "#f5f4ee", "--theme.secondaryBackgroundColor", "#ffffff",
        ], cwd=root)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(cli())
