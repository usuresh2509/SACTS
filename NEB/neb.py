import os
import numpy as np
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS
from xtb.ase.calculator import XTB  # Swap with MACE, DFTB+, etc., if desired

def run_neb():
    # 1. Load the endpoints
    if not os.path.exists('reactant.xyz') or not os.path.exists('product.xyz'):
        raise FileNotFoundError("reactant.xyz or product.xyz missing from the current directory.")
        
    reactant = read('reactant.xyz')
    product = read('product.xyz')

    # 2. Define the number of intermediate images
    num_images = 10 
    
    # Create the initial list of images: [Reactant, Image1, Image2, ..., Product]
    images = [reactant]
    for _ in range(num_images):
        images.append(reactant.copy())
    images.append(product)

    # 3. Initialize NEB and interpolate
    neb = NEB(images, method='improvedtangent')
    neb.interpolate()

    # 4. Attach the calculator to ALL images
    # The endpoints need calculators too so we can evaluate the final energy profile
    for image in images:
        image.calc = XTB(method="GFN2-xTB") 

    # 5. Optimize the NEB path
    trajectory_file = 'neb_path.traj'
    optimizer = BFGS(neb, trajectory=trajectory_file)
    
    print(f"Starting NEB optimization with {num_images} intermediate images...")
    optimizer.run(fmax=0.05) 

    # 6. Extract energies and isolate the Transition State (TS)
    energies = [image.get_potential_energy() for image in images]
    ts_index = np.argmax(energies)
    ts_image = images[ts_index]
    
    # Calculate barriers relative to the reactant (image 0)
    f_barrier = energies[ts_index] - energies[0]
    r_barrier = energies[ts_index] - energies[-1]

    # 7. Write output files
    # Save the full path
    write('neb_optimized_path.xyz', images)
    # Save ONLY the transition state
    write('transition_state.xyz', ts_image)
    
    # Print summary information to the terminal
    print("\n" + "="*40)
    print("NEB CALCULATION COMPLETE")
    print("="*40)
    print(f"Highest energy found at Image {ts_index} (0-indexed).")
    print(f"Forward Activation Energy:  {f_barrier:.4f} eV")
    print(f"Reverse Activation Energy:  {r_barrier:.4f} eV")
    print("-"*40)
    print(f"Full path saved to:         'neb_optimized_path.xyz'")
    print(f"Transition state saved to:  'transition_state.xyz'")
    print("="*40)

if __name__ == "__main__":
    run_neb()