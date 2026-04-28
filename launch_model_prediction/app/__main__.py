"""Application entry point."""
import sys

from app.cli import app as typer_app
from app.tui_app import LunchTUIApp

__all__ = ["app"]


def _no_user_args() -> bool:
    """Return True when invoked as bare ``python -m app`` / ``uv run python -m app``."""
    # Typical sys.argv:
    # python -m app        -> ['.../python', '-m', 'app']
    # python -m app status -> ['.../python', '-m', 'app', 'status']
    # python app/__main__.py -> ['.../python', 'app/__main__.py']
    if len(sys.argv) <= 1:
        return True

    args = sys.argv[1:]
    # Remove a leading '-m app' pair if present.
    if len(args) >= 2 and args[0] == "-m" and args[1] == "app":
        args = args[2:]
    # Also handle direct execution of __main__.py (e.g. ``python app/__main__.py``)
    elif len(args) == 1 and args[0].endswith("__main__.py"):
        args = []

    return not args


def main() -> None:
    """Dispatch to TUI or CLI depending on arguments."""
    if _no_user_args():
        LunchTUIApp().run()
    else:
        typer_app()


if __name__ == "__main__":
    main()
