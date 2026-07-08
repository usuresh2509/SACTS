#!/usr/bin/env python3
"""
Optimize reactant.xyz and product.xyz across reaction folders using GFN2-xTB.
Outputs are saved to an external 'neb_results' directory.

Usage:
    python opt.py
    python opt.py {folder_name} {charge} {multiplicity}
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Force absolute path resolution to survive SLURM environments cleanly
HERE = Path(__file__).resolve().parent
# Steps up to 'paper' directory, then looks for 'reactions'
REACTIONS_DIR = HERE.parent / "reactions"
# Steps up to 'paper' directory, then creates 'neb_results'
RESULTS_DIR = HERE.parent / "neb_results"


def parse_charge_multiplicity(xyz_path: Path) -> tuple[int, int]:
    """Parses charge and multiplicity from the second line of an XYZ file."""
    try:
        with open(xyz_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return 0, 1
        
        comment_line = lines[1].strip()
        
        chrg_match = re.search(r"charge\s*=\s*(-?\d+)", comment_line, re.IGNORECASE)
        mult_match = re.search(r"(mult|multiplicity)\s*=\s*(\d+)", comment_line, re.IGNORECASE)
        
        if chrg_match and mult_match:
            return int(chrg_match.group(1)), int(mult_match.group(2))
        
        numbers = re.findall(r"-?\d+", comment_line)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
            
    except Exception as e:
        print(f"Warning: Could not parse properties from {xyz_path.name}: {e}")
        
    print(f"Warning: Defaulting to charge=0, multiplicity=1 for {xyz_path.name}")
    return 0, 1


def optimize_file(xyz_path: Path, output_path: Path, charge: int, uhf: int, opt_level: str = "normal") -> None:
    """Optimizes an XYZ file using xTB inside an isolated temporary directory."""
    with tempfile.TemporaryDirectory(prefix="xtbopt_") as tmp:
        tmp_path = Path(tmp)
        local_input = tmp_path / "input.xyz"
        shutil.copy(xyz_path, local_input)

        cmd = [
            "xtb", local_input.name,
            "--gfn", "2",
            "--opt", opt_level,
            "--chrg", str(charge),
            "--uhf", str(uhf),
        ]

        result = subprocess.run(
            cmd,
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        optimized = tmp_path / "xtbopt.xyz"
        if result.returncode != 0 or not optimized.is_file():
            print(result.stdout)
            raise RuntimeError(f"xTB failed for {xyz_path.name} (Exit code {result.returncode})")

        if "GEOMETRY OPTIMIZATION CONVERGED" not in result.stdout:
            print(result.stdout)
            raise RuntimeError(f"Optimization for {xyz_path.name} did not converge.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(optimized, output_path)


def process_reaction_folder(folder_path: Path, cmd_charge: int = None, cmd_mult: int = None) -> None:
    """Handles the optimization workflow for a single reaction subfolder."""
    print(f"\n{'='*60}\nProcessing Folder: {folder_path.name}\n{'='*60}")
    
    reactant_in = folder_path / "reactant.xyz"
    product_in = folder_path / "product.xyz"

    if not reactant_in.is_file() or not product_in.is_file():
        print(f"Skipping {folder_path.name}: Missing reactant.xyz or product.xyz")
        return

    if cmd_charge is not None and cmd_mult is not None:
        charge, multiplicity = cmd_charge, cmd_mult
        print(f"Using CLI arguments -> Charge: {charge}, Multiplicity: {multiplicity}")
    else:
        charge, multiplicity = parse_charge_multiplicity(reactant_in)
        print(f"Parsed from file -> Charge: {charge}, Multiplicity: {multiplicity}")

    uhf = max(0, multiplicity - 1)

    reaction_results_dir = RESULTS_DIR / folder_path.name
    reactant_out = reaction_results_dir / "reactant.xyz"
    product_out = reaction_results_dir / "product.xyz"

    try:
        print(f"--> Optimizing Reactant...")
        optimize_file(reactant_in, reactant_out, charge, uhf)
        print(f"--> Optimizing Product...")
        optimize_file(product_in, product_out, charge, uhf)
        print(f"Success! Saved to: {reaction_results_dir}")
    except Exception as e:
        print(f"Error processing {folder_path.name}: {e}")


def main() -> int:
    print(f"DEBUG: Script location is: {HERE}")
    print(f"DEBUG: Looking for 'reactions' directory at: {REACTIONS_DIR}")
    print(f"DEBUG: Output will be saved to: {RESULTS_DIR}\n")

    if shutil.which("xtb") is None:
        sys.exit("Error: 'xtb' executable not found on PATH.")

    args = sys.argv[1:]

    # Mode 1: Bulk automated run
    if len(args) == 0:
        if not REACTIONS_DIR.is_dir():
            sys.exit(f"Error: '{REACTIONS_DIR}' directory does not exist or cannot be read.")
        
        target_folders = sorted([d for d in REACTIONS_DIR.iterdir() if d.is_dir()])
        if not target_folders:
            sys.exit(f"No subfolders found inside {REACTIONS_DIR}")

        for folder in target_folders:
            process_reaction_folder(folder)

    # Mode 2: Targeted single folder run
    elif len(args) == 3:
        folder_name, charge_str, mult_str = args
        target_folder = REACTIONS_DIR / folder_name
        
        if not target_folder.is_dir():
            sys.exit(f"Error: Target folder '{target_folder}' does not exist under {REACTIONS_DIR}")
        
        try:
            charge = int(charge_str)
            multiplicity = int(mult_str)
        except ValueError:
            sys.exit("Error: Charge and Multiplicity must be integers.")

        process_reaction_folder(target_folder, cmd_charge=charge, cmd_mult=multiplicity)

    else:
        sys.exit("Usage error.\nRun all: python opt.py\nRun single: python opt.py {folder_name} {charge} {multiplicity}")

    print("\nExecution finalized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())