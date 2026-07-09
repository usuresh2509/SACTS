import numpy as np
import subprocess
import os
import re
from geometric.molecule import Molecule
from geometric.internal import PrimitiveInternalCoordinates, Distance, Angle, Dihedral

ANG_TO_BOHR = 1.8897261245650618

class ZMatrixParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.atoms = []
        self.coords = []
        self.topology_objs = [] 
        self._parse()

    def _parse(self):
        with open(self.filepath, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        
        start_idx = 0
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start_idx = i + 1
                break
        
        body = lines[start_idx:]
        self.atoms = []
        xyz = [] 

        for i, line in enumerate(body):
            parts = line.split()
            atom_sym = parts[0]
            self.atoms.append(atom_sym)
            
            if i == 0:
                xyz.append(np.array([0.0, 0.0, 0.0]))
            elif i == 1:
                ref1 = int(parts[1]) - 1
                dist = float(parts[2])
                self.topology_objs.append(Distance(i, ref1))
                new_pos = xyz[ref1] + np.array([0.0, 0.0, dist])
                xyz.append(new_pos)
            elif i == 2:
                ref1 = int(parts[1]) - 1
                dist = float(parts[2])
                ref2 = int(parts[3]) - 1
                angle_deg = float(parts[4])
                self.topology_objs.append(Distance(i, ref1))
                self.topology_objs.append(Angle(i, ref1, ref2))
                new_pos = self._nerf(xyz[ref1], xyz[ref2], xyz[ref2]+np.array([0,1,0]), dist, np.radians(180.0 - angle_deg), 0.0)
                xyz.append(new_pos)
            else:
                ref1 = int(parts[1]) - 1
                dist = float(parts[2])
                ref2 = int(parts[3]) - 1
                angle_deg = float(parts[4])
                ref3 = int(parts[5]) - 1
                dihed_deg = float(parts[6])
                self.topology_objs.append(Distance(i, ref1))
                self.topology_objs.append(Angle(i, ref1, ref2))
                self.topology_objs.append(Dihedral(i, ref1, ref2, ref3))
                new_pos = self._nerf(xyz[ref1], xyz[ref2], xyz[ref3], dist, np.radians(180.0 - angle_deg), np.radians(dihed_deg - 90.0))
                xyz.append(new_pos)

        self.coords = np.array(xyz)

    def _nerf(self, a, b, c, r, theta, phi):
        AB = b - a
        BC = c - b
        v_axis = AB / np.linalg.norm(AB)
        n = np.cross(BC, v_axis)
        if np.linalg.norm(n) < 1e-3: n = np.array([0.0, 1.0, 0.0])
        n /= np.linalg.norm(n)
        nxv = np.cross(n, v_axis)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = -r * np.cos(theta) 
        M = np.column_stack((n, nxv, v_axis))
        d_local = np.array([x, y, z])
        d_global = a + M @ d_local
        return d_global

def get_geometry_objects(coords_flat=None, atoms=None, filepath=None, zmat_obj=None):
    if zmat_obj is not None:
        atoms = zmat_obj.atoms
        if coords_flat is None: coords_flat = zmat_obj.coords.flatten()
        coords_ang = coords_flat.reshape(-1, 3)
        temp_filename = "temp_geo_setup.xyz"
        write_xyz(coords_ang, atoms, filename=temp_filename)
        mol = Molecule(temp_filename)
        coords = PrimitiveInternalCoordinates(mol, connect=False)
        coords.Internals = zmat_obj.topology_objs
        return mol, coords

    if filepath and os.path.exists(filepath):
        mol_ref = Molecule(filepath)
        atoms = mol_ref.elem
        if coords_flat is None: coords_flat = mol_ref.xyzs[0].flatten()
    raise ValueError("Fallback XYZ loading not supported in Z-Mat mode.")

def analyze_pes_point(x_flat: np.ndarray, coord_obj, mol_obj, target_vec=None, work_dir="."):
    xyz_filename = os.path.join(work_dir, "calc_temp.xyz")
    coords_ang = x_flat.reshape(-1, 3)
    write_xyz(coords_ang, mol_obj.elem, filename=xyz_filename)

    energy, grad_ang, grad_bohr, hess_bohr = run_xtb(xyz_filename, work_dir=work_dir)
    vib_path = os.path.join(work_dir, "vibspectrum")
    raw_imag_count = parse_xtb_vibspectrum(vib_path)
    
    x_bohr = x_flat * ANG_TO_BOHR
    q = coord_obj.calculate(x_flat) 
    g_internal = coord_obj.calcGrad(x_flat, grad_ang)
    H_internal = coord_obj.calcHess(x_bohr, grad_bohr, hess_bohr)

    eigvals, evecs = np.linalg.eigh(H_internal)

    if target_vec is not None:
        t_norm = target_vec / (np.linalg.norm(target_vec) + 1e-6)
        overlaps = []
        for i in range(evecs.shape[1]):
            vec = evecs[:, i]
            overlap = np.dot(vec, t_norm)
            overlaps.append(abs(overlap))
        imin = np.argmax(overlaps)
    else:
        imin = np.argmin(eigvals)

    v_chosen = evecs[:, imin]
    chosen_curvature = eigvals[imin]

    Ginv = coord_obj.GInverse(x_bohr.reshape(-1, 3))
    w, U = np.linalg.eigh(0.5 * (Ginv + Ginv.T))
    G = U @ np.diag(1.0 / np.clip(w, 1e-12, None)) @ U.T

    norm_factor = np.sqrt(v_chosen @ (G @ v_chosen))
    v_chosen = v_chosen / norm_factor

    alpha = float(v_chosen @ (G @ g_internal))
    g_para = alpha * v_chosen
    g_perp = g_internal - g_para

    return energy, g_internal, v_chosen, g_perp, q, eigvals, chosen_curvature, raw_imag_count

def run_xtb(xyz_path, uhf=1, work_dir="."):
    cmd = ["xtb", xyz_path, "--hess", "--grad", "--uhf", str(uhf)]
    _ = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=work_dir, encoding="utf-8", errors="ignore", check=False)
    
    grad_file = os.path.join(work_dir, "gradient")
    hess_file = os.path.join(work_dir, "hessian")
    
    energy = xtb_energy(grad_file)
    grad_bohr = xtb_gradient(grad_file).reshape(-1)
    grad_ang = grad_bohr * ANG_TO_BOHR
    hess_bohr = xtb_hessian(hess_file)
    
    return energy, grad_ang, grad_bohr, hess_bohr

def xtb_energy(filename="gradient"):
    with open(filename, 'r') as f:
        for line in f:
            if "SCF energy" in line:
                parts = line.strip().split()
                try:
                    energy_idx = parts.index("energy")
                    return float(parts[energy_idx + 2])
                except (ValueError, IndexError): pass
    return 0.0

def xtb_gradient(filename="gradient"):
    grads = []
    with open(filename, "r") as f:
        capture = False
        for line in f:
            if "$grad" in line:
                capture = True
                continue
            if "$end" in line: break
            if capture:
                if "*" in line: raise ValueError("xTB Gradient Overflow")
                clean_line = re.sub(r'(?<=\d)-', ' -', line.strip())
                parts = clean_line.split()
                if len(parts) == 3:
                    try: grads.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError: continue
    return np.array(grads)

def xtb_hessian(filename="hessian"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    values = []
    for line in lines:
        line = line.strip()
        if line.startswith("$") or not line or "hessian" in line.lower(): continue
        if "*" in line: raise ValueError("xTB Hessian Overflow")
        clean_line = re.sub(r'(?<=\d)-', ' -', line)
        try: values.extend(map(float, clean_line.split()))
        except ValueError: continue
    L = len(values); N = int(L ** 0.5)
    if N * N == L: return np.array(values).reshape(N, N)
    if int((L - 1) ** 0.5) ** 2 == (L - 1): return np.array(values[1:]).reshape(int((L - 1) ** 0.5), int((L - 1) ** 0.5))
    raise ValueError(f"Hessian Size Mismatch (Got {L} elements)")

def write_xyz(coords, atoms, filename="molecule.xyz"):
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("Generated for PES Analysis\n")
        for sym, (x, y, z) in zip(atoms, coords):
            f.write(f"{sym:<2} {x: >12.6f} {y: >12.6f} {z: >12.6f}\n")

def parse_xtb_vibspectrum(filepath="vibspectrum"):
    imaginary_count = 0
    capture = False
    
    if not os.path.exists(filepath): return 0
        
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("$vibrational spectrum"):
                capture = True
                continue
            if line.startswith("$end"): break
            if capture and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        wave_number = float(parts[-3])
                        if wave_number < -10.0: imaginary_count += 1
                    except ValueError: continue
    return imaginary_count