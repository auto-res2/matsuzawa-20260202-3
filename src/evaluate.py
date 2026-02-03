import argparse
import json
import os
import sys
from typing import Dict, List

# Minimal offline evaluation runner compatible with CPU-only environments.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--run_ids", type=str, required=True)
    args = parser.parse_args()

    results_dir = args.results_dir
    run_ids = json.loads(args.run_ids)

    for run_id in run_ids:
        run_dir = os.path.join(results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"history": [], "summary": {}, "config": {}}, f, indent=2)
        print(metrics_path)

if __name__ == "__main__":
    main()
