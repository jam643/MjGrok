"""Entry point: uv run python -m mjgrok [--no-preview]"""

import argparse

from mjgrok.gui.app import MjGrokApp


def main() -> None:
    parser = argparse.ArgumentParser(prog="mjgrok")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Start with the embedded simulation preview disabled (toggle it in the GUI).",
    )
    args = parser.parse_args()
    app = MjGrokApp(preview_enabled=not args.no_preview)
    app.run()


if __name__ == "__main__":
    main()
