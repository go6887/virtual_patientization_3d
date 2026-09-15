"""Compatibility entry point: uv run python src/main.py demo."""

from probe_tracking.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
