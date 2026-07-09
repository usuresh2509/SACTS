import numpy as np
import os
from collections import deque
from config import analyze_pes_point, get_geometry_objects, write_xyz
from geometric.internal import Distance, Angle, Dihedral

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
    'Br': 1.20, 'I': 1.39
}


def measure_zmat_numeric(coords_flat, topology_objs):
    coords = coords_flat.reshape(-1, 3)
    values = []
    for obj in topology_objs:
        values.append(obj.value(coords))
    return np.array(values)


class Environment:
    """
    3-action TS-search environment.

    Action layout (length 3):
        action[0] = steering_act  in [-1, 1]   coefficient for v_steer (perp-grad direction)
        action[1] = speed_act     in [-1, 1]   maps to speed_mult in [0.1, 1.0]
        action[2] = rc_steer_raw  in [-1, 1]   maps to rc_steer in [0, 1], coefficient for v_soft

    Step vector:
        raw_step_vec = (rc_steer * v_soft) + (steering_act * v_steer)
    """

    def __init__(self, reactant_zmat, product_zmat, max_steps):
        self.reactant_zmat = reactant_zmat
        self.product_zmat = product_zmat

        self.mol, self.coord_obj = get_geometry_objects(zmat_obj=reactant_zmat)
        self.atoms = self.mol.elem

        self.start_coords = reactant_zmat.coords.flatten()
        self.target_coords = product_zmat.coords.flatten()
        self.custom_spawn_coords = None

        print("   [Env] Using Manual Z-Matrix Topology from reactant file...")
        self.zmat_topology = reactant_zmat.topology_objs

        self.z_bond_indices = []
        self.z_angle_indices = []
        self.z_dihed_indices = []
        for i, obj in enumerate(self.zmat_topology):
            if isinstance(obj, Distance):  self.z_bond_indices.append(i)
            elif isinstance(obj, Angle):   self.z_angle_indices.append(i)
            elif isinstance(obj, Dihedral): self.z_dihed_indices.append(i)
        print(f"   [Env] Z-Mat Breakdown: {len(self.z_bond_indices)} Bonds, "
              f"{len(self.z_angle_indices)} Angles, {len(self.z_dihed_indices)} Dihedrals.")

        self.target_zmat = measure_zmat_numeric(self.target_coords, self.zmat_topology)
        self.target_internals = self.coord_obj.calculate(self.target_coords)

        self.bond_indices = []
        self.angle_indices = []
        self.dihed_indices = []
        for i, obj in enumerate(self.coord_obj.Internals):
            cls_name = obj.__class__.__name__
            if 'Distance' in cls_name:    self.bond_indices.append(i)
            elif 'Angle' in cls_name:     self.angle_indices.append(i)
            elif 'Dihedral' in cls_name:  self.dihed_indices.append(i)

        # ----- Episode bookkeeping -----
        self.max_steps = max_steps
        self.history_length = 5
        self.feature_size = 8
        self.state_size = self.history_length * self.feature_size
        self.history = deque(maxlen=self.history_length)
        self.current_coords = None
        self.current_step = 0
        self.prev_velocity = None

        self.last_n_neg = 0
        self.prev_grad_norm = None
        self.best_episode_grad = 0.1
        self.discovered_ts_zones = []
        self.prev_dist = None
        self.feature_grad_delta = 0.0

        # ----- Steric "ignore mask" (bonded-pair exclusion for proximity sensor) -----
        start_3d = self.start_coords.reshape(-1, 3)
        diff_s = start_3d[:, np.newaxis, :] - start_3d[np.newaxis, :, :]
        dist_s = np.linalg.norm(diff_s, axis=-1)
        bonded_in_reactant = (dist_s < 1.3)
        target_3d = self.target_coords.reshape(-1, 3)
        diff_t = target_3d[:, np.newaxis, :] - target_3d[np.newaxis, :, :]
        dist_t = np.linalg.norm(diff_t, axis=-1)
        bonded_in_product = (dist_t < 1.3)
        self.ignore_mask = np.logical_or(bonded_in_reactant, bonded_in_product)
        np.fill_diagonal(self.ignore_mask, True)

        # ----- Reaction topology: broken/formed pairs, active atoms -----
        self._scan_reaction_topology()

        # ----- Blindfold (only used for ring-opening early-phase masking) -----
        self.auto_blindfold_indices = self._calculate_blindfold_indices()

    # ------------------------------------------------------------------ #
    # Topology scan: produces broken/formed pairs and active atoms.      #
    # ------------------------------------------------------------------ #
    def _scan_reaction_topology(self):
        start_3d = self.start_coords.reshape(-1, 3)
        target_3d = self.target_coords.reshape(-1, 3)

        dist_R = np.linalg.norm(start_3d[:, np.newaxis, :] - start_3d[np.newaxis, :, :], axis=-1)
        dist_P = np.linalg.norm(target_3d[:, np.newaxis, :] - target_3d[np.newaxis, :, :], axis=-1)
        self._dist_R = dist_R
        self._dist_P = dist_P
        self._delta_dist = np.abs(dist_R - dist_P)

        n_atoms = len(self.atoms)
        thresholds = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(n_atoms):
                r_i = COVALENT_RADII.get(self.atoms[i], 0.76)
                r_j = COVALENT_RADII.get(self.atoms[j], 0.76)
                thresholds[i, j] = (r_i + r_j) * 1.3

        conn_R = dist_R < thresholds
        conn_P = dist_P < thresholds
        broken_mask = (conn_R == True)  & (conn_P == False)
        formed_mask = (conn_R == False) & (conn_P == True)

        self.broken_pairs = set()
        for a1, a2 in zip(*np.where(broken_mask)):
            if a1 < a2: self.broken_pairs.add((int(a1), int(a2)))
        self.formed_pairs = set()
        for a1, a2 in zip(*np.where(formed_mask)):
            if a1 < a2: self.formed_pairs.add((int(a1), int(a2)))

        self.active_pairs = list(self.broken_pairs.union(self.formed_pairs))
        self.active_atoms = set()
        for a1, a2 in self.active_pairs:
            self.active_atoms.update([a1, a2])

        print(f"   [Env] Reaction scan: {len(self.broken_pairs)} broken, "
              f"{len(self.formed_pairs)} formed, {len(self.active_atoms)} active atoms.")

    def _calculate_blindfold_indices(self):
        print("\n   [Env] Scanning Reaction Topology for blindfold...")
        broken_pairs = self.broken_pairs
        formed_pairs = self.formed_pairs
        delta_dist = self._delta_dist
        n_atoms = len(self.atoms)

        is_ring_opening = (len(formed_pairs) == 0 and len(broken_pairs) > 0)

        if not is_ring_opening:
            if len(formed_pairs) > 0 and len(broken_pairs) > 0:
                print("   [Env] Reaction Type: Transfer / Isomerization")
            elif len(formed_pairs) > 0 and len(broken_pairs) == 0:
                print("   [Env] Reaction Type: Pure Association / Addition")
            else:
                print("   [Env] Reaction Type: Conformational Change")
            print("   [Env] -> Target Mask DISABLED. Full Guidance active.")
            return []

        print("   [Env] Reaction Type: Pure Dissociation / Ring Opening")
        print("   [Env] -> Target Mask ENABLED. Protecting Early Transition State.")

        active_atoms_local = set()
        for a1, a2 in broken_pairs:
            active_atoms_local.update([a1, a2])

        break_1, break_2 = list(broken_pairs)[0]
        shift_1 = np.sum(delta_dist[break_1])
        shift_2 = np.sum(delta_dist[break_2])
        anchor_idx = break_1 if shift_1 < shift_2 else break_2

        ACTIVE_THRESHOLD = 0.40
        for i in range(n_atoms):
            if delta_dist[anchor_idx, i] >= ACTIVE_THRESHOLD:
                active_atoms_local.add(i)

        blindfold_indices = []
        for i, obj in enumerate(self.zmat_topology):
            target_atom_idx = getattr(obj, 'a', -1)
            if target_atom_idx != -1 and target_atom_idx not in active_atoms_local:
                blindfold_indices.append(i)

        print(f"   [Env] Mask applied to {len(blindfold_indices)} Spectator Z-Matrix indices.\n")
        return blindfold_indices

    def set_custom_endpoints(self, start_coords, target_coords, mode_text=""):
        self.custom_spawn_coords = start_coords.copy()
        self.target_internals = self.coord_obj.calculate(target_coords)
        self.target_coords = target_coords.copy()
        self._scan_reaction_topology()
        print(f"   [Env] Endpoints Locked: {mode_text}")

    def get_signed_diff(self, target, current):
        diff = target - current
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def calculate_target_vector(self, q_current):
        target_vec = self.target_internals - q_current
        for idx in self.dihed_indices:
            target_vec[idx] = self.get_signed_diff(self.target_internals[idx], q_current[idx])
        return target_vec

    # ------------------------------------------------------------------ #
    # Helper: build the (possibly masked) target vector exactly the same #
    # way step() and config see it.                                      #
    # ------------------------------------------------------------------ #
    def _build_target_vec(self, q_current, current_phase):
        target_vec = self.calculate_target_vector(q_current)
        if current_phase <= 1 and self.auto_blindfold_indices:
            target_vec[self.auto_blindfold_indices] = 0.0
        return target_vec

    def get_features(self, coords, properties):
        energy, grad, v_soft, g_perp, q_current, eigvals = properties[:6]
        if len(properties) > 8:
            v_soft_eigval = float(properties[8])
        else:
            v_soft_eigval = float(np.min(eigvals))

        # --- Gradient magnitude ---
        grad_norm = np.linalg.norm(grad)
        steepness = np.log1p(grad_norm)

        # --- Hessian classifier ---
        n_neg_eig = int(np.sum(eigvals < -1e-1))

        # --- Curvature along the followed mode (compressed) ---
        soft_curvature = 5.0 * np.tanh(v_soft_eigval / 5.0)

        # --- Target vector (consistent masking with v_soft selection) ---
        current_phase = getattr(self, 'current_phase', 0)
        target_vec = self._build_target_vec(q_current, current_phase)
        target_norm = np.linalg.norm(target_vec)
        if target_norm > 1e-6:
            target_dir = target_vec / target_norm
        else:
            target_dir = np.zeros_like(target_vec)

        # --- Mode-target alignment ---
        v_soft_norm = v_soft / (np.linalg.norm(v_soft) + 1e-6)
        alignment = float(np.dot(v_soft_norm, target_dir))

        # --- Slope sensor along the followed mode ---
        slope_sensor = float(np.dot(grad, v_soft) / (grad_norm + 1e-6))

        # --- Distance to target ---
        dist_to_target = float(target_norm)

        # --- Steric proximity (non-bonded only) ---
        coords_3d = coords.reshape(-1, 3)
        diff = coords_3d[:, np.newaxis, :] - coords_3d[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        dangerous_dists = dists[~self.ignore_mask]
        min_dist = np.min(dangerous_dists) if len(dangerous_dists) > 0 else 2.5
        proximity_signal = (0.74 / (min_dist + 1e-6)) ** 6
        proximity_signal = float(np.clip(proximity_signal, 0.0, 5.0))

        # --- Temporal grad delta ---
        grad_delta_feature = float(getattr(self, 'feature_grad_delta', 0.0))

        return np.array([
            steepness,             # 0
            n_neg_eig,             # 1
            soft_curvature,        # 2
            slope_sensor,          # 3
            alignment,             # 4
            grad_delta_feature,    # 5
            proximity_signal,      # 6
            dist_to_target,        # 7
        ], dtype=np.float64)

    def reset(self):
        self.found_ts = False
        self.current_step = 0
        self.history.clear()
        self.prev_velocity = None
        self.last_n_neg = 0

        self.prev_grad_norm = 0.1
        self.best_episode_grad = 0.1
        self.feature_grad_delta = 0.0

        if self.custom_spawn_coords is not None:
            self.current_coords = self.custom_spawn_coords.copy()
        else:
            self.current_coords = self.start_coords.copy()

        initial_q = self.coord_obj.calculate(self.current_coords)
        target_vec_raw = self._build_target_vec(initial_q, getattr(self, 'current_phase', 0))
        self.prev_dist = float(np.linalg.norm(target_vec_raw))

        try:
            properties = analyze_pes_point(self.current_coords, self.coord_obj, self.mol)
            features = self.get_features(self.current_coords, properties)
            self.last_n_neg = int(features[1])
        except Exception:
            features = np.zeros(self.feature_size)
        for _ in range(self.history_length - 1):
            self.history.append(np.zeros(self.feature_size))
        self.history.append(features)
        return np.concatenate(self.history, axis=0)

    def check_ts_discovery(self, coords, grad_norm):
        current_zmat_vals = measure_zmat_numeric(coords, self.zmat_topology)
        BOND_TOLERANCE  = 0.4
        ANGLE_TOLERANCE = 0.50
        DIHED_TOLERANCE = 0.50

        for i, zone in enumerate(self.discovered_ts_zones):
            saved_zmat_vals = zone['zmat_vals']
            diff_vec = current_zmat_vals - saved_zmat_vals
            for idx in self.z_dihed_indices:
                diff_vec[idx] = (diff_vec[idx] + np.pi) % (2 * np.pi) - np.pi
            is_match_geom = True
            if self.z_bond_indices:
                if np.max(np.abs(diff_vec[self.z_bond_indices])) > BOND_TOLERANCE: is_match_geom = False
            if self.z_angle_indices and is_match_geom:
                if np.max(np.abs(diff_vec[self.z_angle_indices])) > ANGLE_TOLERANCE: is_match_geom = False
            if self.z_dihed_indices and is_match_geom:
                if np.max(np.abs(diff_vec[self.z_dihed_indices])) > DIHED_TOLERANCE: is_match_geom = False
            if is_match_geom:
                if grad_norm < zone['best_grad']:
                    zone['best_grad'] = grad_norm
                    zone['coords'] = coords.copy()
                    zone['zmat_vals'] = current_zmat_vals.copy()
                    return False, True, i
                else:
                    return False, False, i
        new_zone = {
            'coords': coords.copy(),
            'zmat_vals': current_zmat_vals.copy(),
            'best_grad': grad_norm
        }
        self.discovered_ts_zones.append(new_zone)
        return True, True, len(self.discovered_ts_zones) - 1

    def step(self, action, current_phase=0):
        """
        3-action interface:
            action[0] = steering_act
            action[1] = speed_act
            action[2] = rc_steer_raw
        """
        self.current_phase = current_phase
        done = False
        found_zone_id = None

        steering_act  = action[0]
        speed_act     = action[1]
        rc_steer      = (action[2] + 1.0) / 2.0
        speed_mult    = 0.1 + (0.9 * ((speed_act + 1.0) / 2.0))

        # Phase-dependent step caps
        if current_phase == 2:
            self.max_steps = 300
            max_bond  = 0.005 * speed_mult
            max_angle = 0.010 * speed_mult
            max_dihed = 0.020 * speed_mult
        elif current_phase == 1:
            self.max_steps = 400
            max_bond  = 0.005 * speed_mult
            max_angle = 0.010 * speed_mult
            max_dihed = 0.020 * speed_mult
        else:
            self.max_steps = 200
            max_bond  = 0.05 * speed_mult
            max_angle = 0.10 * speed_mult
            max_dihed = 0.30 * speed_mult

        # Steric pre-check
        coords_3d = self.current_coords.reshape(-1, 3)
        diff = coords_3d[:, np.newaxis, :] - coords_3d[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        masked_dists = np.where(self.ignore_mask, 10.0, dists)
        if np.any(masked_dists < 0.74):
            print(f"   >>> CRASH AVOIDED: Steric Clash (Min Dist: {np.min(masked_dists):.2f})")
            dummy_state = np.concatenate(self.history, axis=0)
            return dummy_state, -200.0, True, {'energy': 0.0, 'target_reached': False, 'grad_rms': 0.0}

        # Build target vector with the SAME masking config will use
        q_current = self.coord_obj.calculate(self.current_coords)
        target_vec_raw = self._build_target_vec(q_current, current_phase)
        curr_dist = float(np.linalg.norm(target_vec_raw))
        if curr_dist > 1e-5:
            target_dir = target_vec_raw / curr_dist
        else:
            target_dir = np.zeros_like(target_vec_raw)

        # PES analysis
        try:
            properties = analyze_pes_point(self.current_coords, self.coord_obj, self.mol,
                                           target_vec=target_vec_raw)
            energy, grad, v_soft, g_perp, q_internal, eigvals = properties[:6]
            raw_imag_count = properties[7]
        except Exception as e:
            print(f">>> xTB ERROR: {e}")
            dummy_state = np.concatenate(self.history, axis=0)
            return dummy_state, -200.0, True, {'energy': 0.0, 'target_reached': False, 'grad_rms': 0.0}

        n_dof = max(1, 3 * len(self.atoms) - 6)
        raw_grad_norm = float(np.linalg.norm(grad))
        grad_rms = raw_grad_norm / np.sqrt(n_dof)
        n_neg_eig = int(np.sum(eigvals < -1e-1))

        # v_soft sign-flip toward target
        if np.dot(v_soft, target_dir) < 0:
            v_soft = -v_soft

        # v_steer = unit perpendicular-gradient direction
        if np.linalg.norm(g_perp) < 1e-5:
            v_steer = np.zeros_like(v_soft)
        else:
            v_steer = g_perp / np.linalg.norm(g_perp)

        # ---- 3-action step composition ----
        raw_step_vec = (rc_steer * v_soft) + (steering_act * v_steer)

        # Per-coord-type clamping
        final_internal_step = np.zeros_like(raw_step_vec)
        scale_b, scale_a, scale_d = 1.0, 1.0, 1.0
        if len(self.bond_indices) > 0:
            req_bond = np.max(np.abs(raw_step_vec[self.bond_indices]))
            scale_b = min(1.0, max_bond / max(req_bond, 1e-6))
            final_internal_step[self.bond_indices] = raw_step_vec[self.bond_indices] * scale_b
        if len(self.angle_indices) > 0:
            req_angle = np.max(np.abs(raw_step_vec[self.angle_indices]))
            scale_a = min(1.0, max_angle / max(req_angle, 1e-6))
            final_internal_step[self.angle_indices] = raw_step_vec[self.angle_indices] * scale_a
        if len(self.dihed_indices) > 0:
            req_dihed = np.max(np.abs(raw_step_vec[self.dihed_indices]))
            scale_d = min(1.0, max_dihed / max(req_dihed, 1e-6))
            final_internal_step[self.dihed_indices] = raw_step_vec[self.dihed_indices] * scale_d

        # Internal -> Cartesian
        candidate_coords_obj = self.coord_obj.newCartesian(self.current_coords, final_internal_step)
        candidate_coords = candidate_coords_obj.flatten()
        candidate_q = self.coord_obj.calculate(candidate_coords)

        # Bond sanity
        if len(self.bond_indices) > 0:
            if np.any(candidate_q[self.bond_indices] > 4.0) or np.any(candidate_q[self.bond_indices] < 0.8):
                dummy_state = np.concatenate(self.history, axis=0)
                return dummy_state, -200.0, True, {'energy': 0.0, 'target_reached': False, 'grad_rms': 0.0}

        self.prev_velocity = final_internal_step
        self.current_coords = candidate_coords
        self.current_step += 1

        # ----- Reward shaping -----
        reward = 0.0
        is_ts = False
        reward -= np.linalg.norm(g_perp)

        dist_delta = self.prev_dist - curr_dist
        reward += dist_delta
        self.prev_dist = curr_dist

        if self.prev_grad_norm is not None:
            grad_delta = grad_rms - self.prev_grad_norm
            self.feature_grad_delta = grad_delta
        else:
            grad_delta = 0.0
            self.feature_grad_delta = 0.0

        if n_neg_eig == 0:
            reward -= 0.1 * self.current_step
        elif n_neg_eig == 1:
            if raw_imag_count > 1 or raw_imag_count == 0:
                reward -= 10.0
                reward -= 2.0 * np.linalg.norm(g_perp)
            else:
                reward -= 2.0 * np.linalg.norm(g_perp)
                if grad_rms < self.best_episode_grad:
                    reward += 20.0
                    self.best_episode_grad = grad_rms

                    is_new, is_better, zone_id = self.check_ts_discovery(self.current_coords, grad_rms)
                    if is_new:
                        is_ts = True
                        found_zone_id = zone_id
                        print(f">>> DISCOVERY: New TS Zone {zone_id}! (RMS: {grad_rms:.4f})")
                        reward += 200.0
                    elif is_better:
                        is_ts = True
                        found_zone_id = zone_id
                        print(f">>> UPDATE: TS Zone {zone_id} improved.")
                        reward += 50.0

                if self.prev_grad_norm is not None:
                    if grad_delta > 0:
                        penalty_mult = 50.0 if current_phase > 0 else 200.0
                        reward -= penalty_mult * grad_delta
                    else:
                        reward += 100.0 * abs(grad_delta)
        else:
            reward -= 5.0 * (n_neg_eig - 1)
            reward -= 25.0
            reward -= 10.0 * grad_rms

        self.prev_grad_norm = grad_rms
        self.last_n_neg = n_neg_eig

        if hasattr(self.coord_obj, 'clearCache'):
            self.coord_obj.clearCache()
        elif hasattr(self.coord_obj, '_cache'):
            self.coord_obj._cache = {}

        print(f"Step {self.current_step} | S_b: {scale_b:.2f} S_a: {scale_a:.2f} S_d: {scale_d:.2f} "
              f"| RMS: {grad_rms:.4f} | N: {n_neg_eig} | Rw: {reward:.2f} "
              f"| raw_imag: {raw_imag_count} | d_grad: {self.feature_grad_delta:.5f}")

        # ----- Convergence checks -----
        curr_vec = self.calculate_target_vector(candidate_q)

        if current_phase == 2:
            bond_tol, angle_tol, dihed_tol = 0.02, 0.02, 0.05
            is_min = (grad_rms < 0.0005 and n_neg_eig == 0)
        elif current_phase == 1:
            bond_tol, angle_tol, dihed_tol = 0.10, 0.05, 0.10
            is_min = (grad_rms < 0.001 and n_neg_eig == 0)
        else:
            bond_tol, angle_tol, dihed_tol = 0.20, 0.10, 0.15
            is_min = (grad_rms < 0.005 and n_neg_eig == 0)

        bond_converged = True
        if self.bond_indices:
            bond_converged = (np.sum(np.abs(curr_vec[self.bond_indices])) < bond_tol)
        ang_converged = True
        if self.angle_indices:
            ang_converged = (np.sum(np.abs(curr_vec[self.angle_indices])) < angle_tol)
        dihed_converged = True
        if self.dihed_indices:
            dihed_converged = (np.sum(np.abs(curr_vec[self.dihed_indices])) < dihed_tol)

        is_converged_ts = ((raw_grad_norm < 0.005) and (n_neg_eig == 1) and (raw_imag_count == 1))

        reached = False
        if (bond_converged and ang_converged and dihed_converged) or is_converged_ts:
            if is_converged_ts:
                print(f"   >>> 🏆 SUCCESS: Absolute TS Convergence Reached! (Grad: {raw_grad_norm:.5f})")
                reward += 500.0
            elif is_ts:
                print(f"   >>> 🎯 SUCCESS: Target Geometry Reached (while in TS Zone)!")
                reward += 200.0
            else:
                print(f"   >>> 🎯 SUCCESS: Target Geometry Reached!")
                reward += 50.0
            reached = True
            done = True

        if self.current_step >= self.max_steps:
            done = True

        next_features = self.get_features(self.current_coords, properties)
        self.history.append(next_features)
        next_state = np.concatenate(self.history, axis=0)

        info = {
            'energy': energy,
            'target_reached': reached,
            'is_ts': is_ts,
            'is_min': is_min,
            'ts_zone': found_zone_id,
            'grad_rms': grad_rms,
            'raw_grad': raw_grad_norm,
            'n_neg': n_neg_eig,
            'absolute_convergence': is_converged_ts,
        }
        return next_state, reward, done, info