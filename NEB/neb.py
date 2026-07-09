#!/usr/bin/env python3
"""
Run NEB calculations on optimized endpoints found inside the 'neb_results' folder.
Automatically cleans and overwrites the transition_state.xyz file to a standard vanilla format.

Usage:
    # Mode 1: Automated bulk NEB run over all folders in neb_results/
    python neb.py

    # Mode 2: Targeted NEB run for a specific folder
    python neb.py hcn
"""

import os
import re
import sys
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS
from xtb.ase.calculator import XTB

HERE = Path(__file__).resolve().parent
# neb_results is sitting outside the 'neb' folder, parallel to it
RESULTS_DIR = HERE.parent / "neb_results"


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


def process_neb_for_folder(folder_path: Path) -> None:
    """Runs the NEB path optimization for a single reaction results directory."""
    print(f"\n{'='*60}\nStarting NEB for Reaction: {folder_path.name}\n{'='*60}")
    
    reactant_path = folder_path / "reactant.xyz"
    product_path = folder_path / "product.xyz"

    if not reactant_path.is_file() or not product_path.is_file():
        print(f"Skipping {folder_path.name}: Missing optimized reactant.xyz or product.xyz")
        return

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

    # 3. Initialize NEB and interpolate using IDPP
    neb = NEB(images, method='improvedtangent')
    neb.interpolate()

    # 4. Attach the calculator to ALL images with correct electronic parameters
    for image in images:
        image.calc = XTB(method="GFN2-xTB", chrg=charge, uhf=uhf) 

    # Define output files to write inside the specific reaction folder
    trajectory_file = str(folder_path / 'neb_path.traj')
    optimized_path_file = str(folder_path / 'neb_optimized_path.xyz')
    ts_file = folder_path / 'transition_state.xyz'

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
    print(f"Outputs written to:        {folder_path}")
    print("="*60 + "\n")


def main() -> int:
    print(f"DEBUG: Script location is: {HERE}")
    print(f"DEBUG: Targeting results folder: {RESULTS_DIR}\n")

    if not RESULTS_DIR.is_dir():
        sys.exit(f"Error: '{RESULTS_DIR}' folder does not exist. Run opt.py first.")

    args = sys.argv[1:]

    # Mode 1: Bulk automatic run over all folders inside neb_results
    if len(args) == 0:
        target_folders = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
        if not target_folders:
            sys.exit(f"No subfolders found inside {RESULTS_DIR}")

        for folder in target_folders:
            process_neb_for_folder(folder)

    # Mode 2: Targeted single folder run
    elif len(args) == 1:
        folder_name = args[0]
        target_folder = RESULTS_DIR / folder_name
        
        if not target_folder.is_dir():
            sys.exit(f"Error: Target folder '{target_folder}' does not exist inside neb_results.")
        
        process_neb_for_folder(target_folder)

    else:
        sys.exit("Usage error.\nRun all: python neb.py\nRun single: python neb.py {folder_name}")

    print("All requested NEB calculations completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())