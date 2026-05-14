"""Console entry point for the dashboard."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def main() -> None:
    if find_spec("streamlit") is None:
        message = (
            "Streamlit is not installed. Install the dashboard extra first:\n"
            '  python -m pip install -e ".[dashboard]"'
        )
        raise SystemExit(message)

    app_path = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
