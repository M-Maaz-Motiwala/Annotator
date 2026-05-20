import argparse

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GPU transcription API health.")
    parser.add_argument("--api-url", required=True, help="Base URL, e.g. http://10.0.0.12:8000")
    args = parser.parse_args()

    resp = requests.get(f"{args.api_url}/health", timeout=30)
    resp.raise_for_status()
    print(resp.json())


if __name__ == "__main__":
    main()
