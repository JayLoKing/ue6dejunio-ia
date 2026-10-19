"""Readiness requires a loaded model, not merely an HTTP 200 response."""

import json
import os
import sys
import urllib.request


def main() -> int:
    port = int(os.environ.get("UE6_API_PORT", "8001"))
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
            payload = json.load(response)
        return 0 if payload.get("status") == "ok" else 1
    except (OSError, ValueError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
