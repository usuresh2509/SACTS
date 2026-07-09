#!/usr/bin/env python3
"""
Run Gaussian transition state optimizations using xTB-Gaussian external script 
on 'ts_guess.xyz' files found across your reaction directories.

Usage:
    python berny.py
    python berny.py hcn
"""

import os
import re
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REACTIONS_DIR = HERE.parent / "reactions"
BERNY_RESULTS_DIR = HERE.parent / "berny_results"


def parse_xyz_file(xyz_path: Path) -> tuple[int, int, list[str]]:
    """
    Parses charge, multiplicity from the 2nd line, and extracts 
    the raw atom symbol + coordinate strings.
    """
    atoms_and_coords = []
    try:
        with open(xyz_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 3:
            raise ValueError(f"{xyz_path.name} does not contain enough lines.")
        
        # Parse charge and multiplicity from the second line
        comment_line = lines[1].strip()
        numbers = re.findall(r"-?\d+", comment_line)
        if len(numbers) >= 2:
            charge, multiplicity = int(numbers[0]), int(numbers[1])
        else:
            print(f"Warning: Could not parse metadata from {xyz_path.name}. Defaulting to 0 1.")
            charge, multiplicity = 0, 1
            
        # Parse atom positions (skipping lines 1 and 2)
        for line in lines[2:]:
            line_str = line.strip()
            if line_str:  # Avoid empty lines
                atoms_and_coords.append(line_str)
                
        return charge, multiplicity, atoms_and_coords
        
    except Exception as e:
        print(f"Error parsing file {xyz_path}: {e}")
        sys.exit(1)


def parse_optimized_ts_from_log(log_path: Path, dest_xyz_path: Path) -> bool:
    """
    Parses the final optimized coordinates from a Gaussian standard log file
    when an optimization completes, and saves it as a clean vanilla xyz structure.
    """
    if not log_path.is_file():
        return False
        
    with open(log_path, "r") as f:
        log_content = f.read()

    # Check for normal termination and convergence criteria met
    if "Stationary point found." not in log_content:
        print(f"Warning: 'Stationary point found.' not verified in {log_path.name}. Structure might not be fully converged.")

    # Locate the last coordinate block print out from Standard orientation
    # We look for Standard orientation blocks backwards
    with open(log_path, "r") as f:
        lines = f.readlines()

    coord_start_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Standard orientation:" in lines[i]:
            coord_start_idx = i
            break
            
    if coord_start_idx == -1:
        return False

    # Skip header lines to reach atomic positions
    # Line index mapping:
    # coord_start_idx -> "Standard orientation:"
    # +1 -> "---------------------------------------------------------------------"
    # +2 -> "Center     Atomic      Atomic             Coordinates (Angstroms)"
    # +3 -> "Number     Number       Type             X           Y           Z"
    # +4 -> "---------------------------------------------------------------------"
    # +5 -> First line of coordinate content
    
    atoms = []
    # Periodic table dictionary mapping for converting atomic numbers back to symbols
    periodic_table = {
        1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"
    }

    for line in lines[coord_start_idx + 5:]:
        if "---------------------------------------------------------------------" in line:
            break
        parts = line.split()
        if len(parts) == 6:
            atomic_num = int(parts[1])
            element_sym = periodic_table.get(atomic_num, f"X{atomic_num}")
            x, y, z = parts[3], parts[4], parts[5]
            atoms.append(f"{element_sym:<4} {x:>12} {y:>12} {z:>12}")

    if not atoms:
        return False

    # Write out a clean, standardized, vanilla XYZ file
    with open(dest_xyz_path, "w") as xyz_out:
        xyz_out.write(f"{len(atoms)}\n")
        xyz_out.write("\n")  # Blank clean comment line
        for atom_line in atoms:
            xyz_out.write(f"{atom_line}\n")
            
    return True


def process_reaction(folder_path: Path) -> None:
    """Creates input files, hooks subprocess to g16, and processes results for Berny TS."""
    print(f"\n{'='*60}\nRunning Berny Optimization for: {folder_path.name}\n{'='*60}")
    
    ts_guess_file = folder_path / "ts_guess.xyz"
    if not ts_guess_file.is_file():
        print(f"Skipping {folder_path.name}: 'ts_guess.xyz' not found.")
        return

    # 1. Gather properties and structures dynamically
    charge, multiplicity, atom_coordinates = parse_xyz_file(ts_guess_file)
    print(f"Loaded properties -> Charge: {charge}, Multiplicity: {multiplicity}")

    # Build the specialized .gjf / .com string format
    coordinate_block = "\n".join(atom_coordinates)
    gjf_content = (
        f"%nprocshared=2\n"
        f"%mem=2GB\n"
        f"%chk={folder_path.name}.chk\n"
        f'#p External="/lustre/isaac24/scratch/usuresh3/xtb-gaussian/xtb-g" Opt=(TS, CalcFC, NoEigenTest, NoMicro)\n'
        f"\n"
        f"Title: {folder_path.name} ts search via xTB-Gaussian\n"
        f"\n"
        f"{charge} {multiplicity}\n"
        f"{coordinate_block}\n"
        f"\n"
    )

    # 2. Setup the output folder directory hierarchy
    reaction_results_dir = BERNY_RESULTS_DIR / folder_path.name
    reaction_results_dir.mkdir(parents=True, exist_ok=True)
    
    com_file_path = reaction_results_dir / f"{folder_path.name}.com"
    log_file_path = reaction_results_dir / f"{folder_path.name}.log"
    ts_final_path = reaction_results_dir / "ts.xyz"

    # Write input instruction file directly inside the target output subfolder
    with open(com_file_path, "w") as f:
        f.write(gjf_content)

    print(f"Generated input setup file inside {reaction_results_dir.name}/")
    print("Launching Gaussian 16 workspace execution package...")

    # 3. Call g16 using subprocess context handles directly in that folder
    try:
        with open(log_file_path, "w") as log_file:
            subprocess.run(
                ["g16"], 
                stdin=open(com_file_path, "r"), 
                stdout=log_file, 
                cwd=reaction_results_dir,  # Directs scratch runtime .chk allocations locally
                check=True
            )
        print("Gaussian core calculation completed successfully.")
        
        # 4. Extract finalized geometry immediately post calculations
        print("Extracting final stationary optimization geometry into standard format...")
        success = parse_optimized_ts_from_log(log_file_path, ts_final_path)
        if success:
            print(f"Success! Clean optimized TS written to: {ts_final_path.name}")
        else:
            print("Warning: Completed calculation, but could not parse coordinates out of the standard orientation blocks.")
            
    except subprocess.CalledProcessError as err:
        print(f"Execution Error: Gaussian process failed with error code context tracking: {err}")
    except Exception as general_err:
        print(f"An unexpected script handling exception occurred: {general_err}")


def main() -> int:
    if shutil.which("g16") is None:
        sys.exit("Error: 'g16' command line utility execution pathway was not found on active system PATH.")

    args = sys.argv[1:]

    # Mode 1: Automated scan across all folders in reactions/
    if len(args) == 0:
        if not REACTIONS_DIR.is_dir():
            sys.exit(f"Error: Target reactions scan base configuration point path '{REACTIONS_DIR}' does not exist.")
            
        target_folders = sorted([d for d in REACTIONS_DIR.iterdir() if d.is_dir()])
        if not target_folders:
            sys.exit(f"No valid chemical structure directories detected under route directory context tracking: {REACTIONS_DIR}")

        for folder in target_folders:
            process_reaction(folder)

    # Mode 2: Targeted single reaction workspace calculation 
    elif len(args) == 1:
        folder_name = args[0]
        target_folder = REACTIONS_DIR / folder_name
        
        if not target_folder.is_dir():
            sys.exit(f"Error: Targeted chemical directory tracking label path '{target_folder}' could not be matched under base hierarchy context point.")
            
        process_reaction(target_folder)

    else:
        sys.exit("Usage error tracking arguments layout.\nRun bulk updates: python berny.py\nRun single track step: python berny.py {folder_name}")

    print("\nAll execution schedules finalized safely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())