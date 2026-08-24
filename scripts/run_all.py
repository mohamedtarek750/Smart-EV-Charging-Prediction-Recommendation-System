"""One-command pipeline: data -> features -> models -> figures -> tests.

    python -m scripts.run_all              full rebuild
    python -m scripts.run_all --skip-data  reuse the committed dataset
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("simulate the network", "src.data.simulate", "data"),
    ("build features", "src.features.build_features", "features"),
    ("train models", "src.models.train", "models"),
    ("render figures", "scripts.make_figures", "figures"),
    ("run tests", "tests.test_pipeline", "tests"),
]


def run(module: str) -> int:
    return subprocess.call([sys.executable, "-m", module])


def main() -> int:
    ap = argparse.ArgumentParser()
    for _, _, tag in STEPS:
        ap.add_argument(f"--skip-{tag}", action="store_true")
    args = ap.parse_args()

    for label, module, tag in STEPS:
        if getattr(args, f"skip_{tag}"):
            print(f"\n>>> skipping {label}")
            continue
        print(f"\n{'=' * 64}\n>>> {label}  ({module})\n{'=' * 64}")
        t0 = time.time()
        code = run(module)
        print(f"--- {label}: {'ok' if code == 0 else f'FAILED ({code})'} in {time.time()-t0:.1f}s")
        if code != 0:
            return code
    print("\npipeline complete. Next:\n"
          "  streamlit run app/dashboard.py\n"
          "  uvicorn app.api:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
