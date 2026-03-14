"""Entry point: uv run mjpython -m mjgrok"""

from mjgrok.gui.app import MjGrokApp


def main() -> None:
    app = MjGrokApp()
    app.run()


if __name__ == "__main__":
    main()
