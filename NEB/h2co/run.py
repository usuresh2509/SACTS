#!/usr/bin/env python3
"""
Optimize reactant.xyz and product.xyz with GFN2-xTB and overwrite the
originals with the relaxed geometries.

Each file is optimized in its own scratch directory so xtb's output files
(xtbopt.xyz, xtbopt.log, etc.) don't collide or clutter the run folder.

Usage:
    python optimize_endpoints.py
    python optimize_endpoints.py --charge 0 --uhf 0 --opt-level tight
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = ["reactant.xyz", "product.xyz"]


def optimize(xyz_path: Path, charge: int, uhf: int, opt_level: str) -> None:
    if not xyz_path.is_file():
        raise FileNotFoundError(f"{xyz_path.name} not found in {xyz_path.parent}")

    with tempfile.TemporaryDirectory(prefix="xtbopt_") as tmp:
        tmp = Path(tmp)
        local_input = tmp / "input.xyz"
        shutil.copy(xyz_path, local_input)

        cmd = [
            "xtb", local_input.name,
            "--gfn", "2",
            "--opt", opt_level,
            "--chrg", str(charge),
            "--uhf", str(uhf),
        ]

        print(f"[{xyz_path.name}] running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        optimized = tmp / "xtbopt.xyz"
        if result.returncode != 0 or not optimized.is_file():
            print(result.stdout)
            raise RuntimeError(
                f"xtb failed for {xyz_path.name} "
                f"(return code {result.returncode}); xtbopt.xyz not produced."
            )

        # Confirm convergence in the xtb log before overwriting.
        if "GEOMETRY OPTIMIZATION CONVERGED" not in result.stdout:
            print(result.stdout)
            raise RuntimeError(
                f"Optimization for {xyz_path.name} did not report convergence. "
                f"Original file left untouched."
            )

        shutil.copy(optimized, xyz_path)
        print(f"[{xyz_path.name}] converged -> overwrote original\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="GFN2-xTB optimize endpoints in place.")
    parser.add_argument("--charge", type=int, default=0, help="total charge (default 0)")
    parser.add_argument("--uhf", type=int, default=0,
                        help="number of unpaired electrons (default 0)")
    parser.add_argument("--opt-level", default="normal",
                        help="xtb opt level: crude/sloppy/loose/normal/tight/vtight/extreme")
    args = parser.parse_args()

    if shutil.which("xtb") is None:
        sys.exit("Error: 'xtb' executable not found on PATH.")

    for name in TARGETS:
        optimize(HERE / name, args.charge, args.uhf, args.opt_level)

    print("Done. Both files optimized and rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())