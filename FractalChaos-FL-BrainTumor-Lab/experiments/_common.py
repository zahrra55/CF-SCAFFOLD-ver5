import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcfl.config import OUTPUT_DIR  # noqa: E402

FINAL_SET_FILE = os.path.join(OUTPUT_DIR, "final_descriptor_set.json")


def heading(text):
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78)


def report_test(name, m, p, ci):
    print(f"  {name:56s} {100 * m:+6.2f} pp  [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]  p = {p:.4f}")


def load_final_set():
    if not os.path.exists(FINAL_SET_FILE):
        raise SystemExit(
            "Run experiments/01_descriptor_classification.py first: it selects the descriptor set."
        )
    with open(FINAL_SET_FILE) as f:
        return json.load(f)


def save_final_set(info):
    with open(FINAL_SET_FILE, "w") as f:
        json.dump(info, f, indent=1)
