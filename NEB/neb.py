#!/usr/bin/env python3
"""
Run NEB calculations on optimized endpoints found inside the 'neb_results' folder.
Outputs are saved to either 'neb_results_linear' or 'neb_results_idpp' depending 
on the chosen interpolation method.
Automatically cleans and overwrites the transition_state.xyz file to a standard vanilla format.

Usage:
    # Mode 1: Automated bulk NEB run
    python neb.py --method linear
    python neb.py --method idpp

    # Mode 2: Targeted NEB run for a specific folder
    python neb.py hcn --method linear
"""

import os
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS
from xtb.ase.calculator import XTB

HERE = Path(__file__).resolve().parent
# Source directory containing the optimized endpoints
INPUT_DIR = HERE.parent / "neb_results"


def parse_charge_multiplicity(xyz_path: Path) -> tuple[int, int]:
    """Parses raw charge and multiplicity integers from the second line of an XYZ file."""
    try:
        with open(xyz_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return 0, 1
        
        comment_line = lines[1].strip()
        numbers = re.findall(r"-?\d+", comment_line)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
            
    except Exception as e:
        print(f"Warning: Could not read metadata from {xyz_path.name}: {e}")
        
    print(f"Warning: Defaulting to charge=0, multiplicity=1 for {xyz_path.name}")
    return 0, 1


def clean_xyz_inplace(file_path: Path) -> None:
    """
    Reads an Extended XYZ file and overwrites it using the plain 'xyz' format,
    stripping extended properties, forces, charges, dipoles, etc.
    """
    if not file_path.is_file():
        print(f"Error: {file_path.name} not found for cleaning.")
        return

    # Read the image frame(s)
    images = read(file_path, index=':')
    
    # Write back to the same path using the plain 'xyz' format specification
    write(file_path, images, format='xyz')
    print(f"--> Cleaned extended properties from '{file_path.name}' successfully (overwritten).")


def process_neb_for_folder(folder_path: Path, method: str) -> None:
    """Runs the NEB path optimization for a single reaction results directory."""
    print(f"\n{'='*60}\nStarting NEB for Reaction: {folder_path.name} (Method: {method.upper()})\n{'='*60}")
    
    reactant_path = folder_path / "reactant.xyz"
    product_path = folder_path / "product.xyz"

    if not reactant_path.is_file() or not product_path.is_file():
        print(f"Skipping {folder_path.name}: Missing optimized reactant.xyz or product.xyz")
        return

    # Create the targeted output directory
    output_base_dir = HERE.parent / f"neb_results_{method}"
    out_folder = output_base_dir / folder_path.name
    out_folder.mkdir(parents=True, exist_ok=True)

    # Extract charge and multiplicity from our modified format (e.g. "0 1")
    charge, multiplicity = parse_charge_multiplicity(reactant_path)
    uhf = max(0, multiplicity - 1)
    print(f"Parsed parameters -> Charge: {charge}, Multiplicity: {multiplicity} (UHF: {uhf})")

    # 1. Load the endpoints
    reactant = read(str(reactant_path))
    product = read(str(product_path))

    # 2. Define the number of intermediate images
    num_images = 10 
    
    # Create the initial list of images: [Reactant, Image1, Image2, ..., Product]
    images = [reactant]
    for _ in range(num_images):
        images.append(reactant.copy())
    images.append(product)

    # 3. Initialize NEB and apply the chosen interpolation method
    neb = NEB(images, method='improvedtangent')
    if method == "idpp":
        neb.interpolate('idpp')
    elif method == "linear":
        neb.interpolate()  # Empty defaults to linear interpolation

    # 4. Attach the calculator to ALL images with correct electronic parameters
    for image in images:
        image.calc = XTB(method="GFN2-xTB", chrg=charge, uhf=uhf) 

    # Define output files to write inside the specific method folder
    trajectory_file = str(out_folder / 'neb_path.traj')
    optimized_path_file = str(out_folder / 'neb_optimized_path.xyz')
    ts_file = out_folder / 'transition_state.xyz'

    # 5. Optimize the NEB path
    optimizer = BFGS(neb, trajectory=trajectory_file)
    
    print(f"Running path optimization (fmax=0.05)...")
    optimizer.run(fmax=0.05) 

    # 6. Extract energies and isolate the Transition State (TS)
    energies = [image.get_potential_energy() for image in images]
    ts_index = np.argmax(energies)
    ts_image = images[ts_index]
    
    # Calculate barriers relative to the reactant (image 0)
    f_barrier = energies[ts_index] - energies[0]
    r_barrier = energies[ts_index] - energies[-1]

    # 7. Write output files to their respective reaction folders
    write(optimized_path_file, images)
    write(str(ts_file), ts_image)
    
    # Automatically clean the transition state file in place right after generation
    clean_xyz_inplace(ts_file)
    
    # Print summary information to the terminal/log file
    print("\n" + "-"*40)
    print(f"NEB COMPLETE FOR: {folder_path.name}")
    print("-"*40)
    print(f"Highest energy found at Image {ts_index} (0-indexed).")
    print(f"Forward Activation Energy:  {f_barrier:.4f} eV")
    print(f"Reverse Activation Energy:  {r_barrier:.4f} eV")
    print(f"Outputs written to:        {out_folder}")
    print("="*60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NEB calculations on optimized endpoints.")
    parser.add_argument("folder", nargs="?", default=None, 
                        help="Specific folder inside neb_results to run (optional).")
    parser.add_argument("--method", choices=["linear", "idpp"], default="idpp", 
                        help="Interpolation method: 'linear' or 'idpp' (default: idpp).")
    args = parser.parse_args()

    print(f"DEBUG: Script location is: {HERE}")
    print(f"DEBUG: Targeting source folder: {INPUT_DIR}")
    print(f"DEBUG: Selected interpolation method: {args.method.upper()}\n")

    if not INPUT_DIR.is_dir():
        sys.exit(f"Error: '{INPUT_DIR}' folder does not exist. Run opt.py first.")

    # Mode 1: Targeted single folder run
    if args.folder:
        target_folder = INPUT_DIR / args.folder
        if not target_folder.is_dir():
            sys.exit(f"Error: Target folder '{target_folder}' does not exist inside neb_results.")
        process_neb_for_folder(target_folder, args.method)

    # Mode 2: Bulk automatic run over all folders inside neb_results
    else:
        target_folders = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
        if not target_folders:
            sys.exit(f"No subfolders found inside {INPUT_DIR}")

        for folder in target_folders:
            process_neb_for_folder(folder, args.method)

    print("All requested NEB calculations completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())