"""Entry point for the TUI: python -m inferall.tui"""

import argparse
from inferall.tui.app import run_dashboard


def main():
    parser = argparse.ArgumentParser(description="InferAll Dashboard")
    parser.add_argument(
        "--url", "-u",
        default="http://127.0.0.1:8000",
        help="Server URL (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()
    run_dashboard(server_url=args.url)


if __name__ == "__main__":
    main()
