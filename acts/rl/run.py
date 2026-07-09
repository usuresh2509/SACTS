import numpy as np
import os
import sys
import shutil
import glob
import subprocess
import argparse

import torch
from geometric.molecule import Molecule
from env import Environment
from agent import SACAgent, ReplayBuffer
from config import get_geometry_objects, write_xyz, ZMatrixParser, run_xtb


def save_dynamic_trajectory(trajectory_slice, save_dir, atoms, ts_relative_index, best_grms=None, best_g=None):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print(f"   >>> SAVING DYNAMIC IRC TRAJECTORY ({len(trajectory_slice)} frames) to {save_dir}")
    for i, coords in enumerate(trajectory_slice):
        rel_step = i - ts_relative_index
        if rel_step == 0:
            if best_grms is not None and best_g is not None:
                fname = f"best_grms_{best_grms:.5f}_g_{best_g:.5f}.xyz"
            else:
                fname = "traj_step_0_BEST.xyz"
        else:
            fname = f"traj_step_{rel_step:+d}.xyz"

        full_path = os.path.join(save_dir, fname)
        write_xyz(coords.reshape(-1, 3), atoms, full_path)


# ==========================================
# PER-REACTION TRAINING PIPELINE
# ==========================================
def train_molecule(agent, memory, molecule_dir, mol_name):
    print("\n" + "="*70)
    print(f"   🚀 EVALUATING REACTION: {mol_name.upper()}")
    print("="*70)

    RAW_R_PATH = os.path.join(molecule_dir, "reactant.zmat")
    RAW_P_PATH = os.path.join(molecule_dir, "product.zmat")

    if not os.path.exists(RAW_R_PATH) or not os.path.exists(RAW_P_PATH):
        print(f"   [!] Skipping {mol_name}: Missing reactant or product file.")
        return False

    r_zmat = ZMatrixParser(RAW_R_PATH)
    p_zmat = ZMatrixParser(RAW_P_PATH)
    env = Environment(r_zmat, p_zmat, max_steps=200)

    # --- TRACKING VARIABLES FOR RUN DETAILS ---
    phase0_episodes = 0
    phase0_total_steps = 0
    phase1_episodes = 0
    phase1_total_steps = 0

    try:
        zone_records = {}
        MAX_ZONES = 2

        molecule_best_grad = 999.0
        molecule_best_raw_grad = 999.0
        molecule_best_traj = []
        molecule_best_zone = "Unknown"

        # ==========================================
        # STAGE 1: PHASE 0 — BROAD EXPLORATION
        # ==========================================
        print("\n" + "="*60)
        print("   STAGE 1: PHASE 0 (ISOLATING THE 2 BEST VALLEYS)")
        print("="*60)

        current_phase = 0
        episodes_without_improvement = 0
        PHASE_0_FALLBACK = 30
        global_best_phase0_grad = float('inf')

        for episode in range(250):
            phase0_episodes += 1  # Track Episode
            improved_this_episode = False
            state = env.reset()
            done = False

            episode_history = [{'coords': env.current_coords.copy(), 'n_neg': getattr(env, 'last_n_neg', 0)}]

            while not done:
                phase0_total_steps += 1  # Track Steps
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action, current_phase)

                n_val = info.get('n_neg', 0)
                episode_history.append({'coords': env.current_coords.copy(), 'n_neg': n_val})
                memory.push(state, action, reward, next_state, 1 if done else 0)
                state = next_state

                if info.get('is_ts', False):
                    zone_id          = info.get('ts_zone')
                    current_grad     = info.get('grad_rms', 999.0)
                    current_raw_grad = info.get('raw_grad', 999.0)

                    if current_grad < molecule_best_grad:
                        molecule_best_grad = current_grad
                        molecule_best_raw_grad = current_raw_grad
                        molecule_best_traj = [frame['coords'] for frame in episode_history]
                        molecule_best_zone = str(zone_id)

                    is_tracked = zone_id in zone_records
                    if not is_tracked:
                        if len(zone_records) < MAX_ZONES:
                            is_tracked = True
                        else:
                            worst_zone_id = max(zone_records.keys(), key=lambda k: zone_records[k]['grad'])
                            worst_grad = zone_records[worst_zone_id]['grad']
                            if current_grad < worst_grad:
                                print(f"   [-] DROPPING Zone {worst_zone_id} (RMS: {worst_grad:.5f}) for Zone {zone_id} (RMS: {current_grad:.5f})")
                                del zone_records[worst_zone_id]
                                shutil.rmtree(os.path.join(molecule_dir, f"ts_structures/zone_{worst_zone_id}"), ignore_errors=True)
                                is_tracked = True

                    if is_tracked:
                        existing_grad = zone_records.get(zone_id, {}).get('grad', float('inf'))
                        if current_grad < existing_grad:
                            if zone_id not in zone_records:
                                print(f"   [+] NEW Zone Tracked: Zone {zone_id} (RMS: {current_grad:.5f})")
                            else:
                                print(f"   [!] Phase 0: Zone {zone_id} Improved! RMS: {current_grad:.5f}")

                            improved_this_episode = True
                            current_idx = len(episode_history) - 1

                            back_idx = current_idx
                            while back_idx > 0 and episode_history[back_idx]['n_neg'] != 0:
                                back_idx -= 1

                            if back_idx == 0 and episode_history[0]['n_neg'] != 0:
                                spawn_coords = env.start_coords.copy()
                                print(f"   [!] No n_neg=0 found — spawning from reactant.")
                            else:
                                spawn_coords = episode_history[back_idx]['coords'].copy()

                            forward_idx = min(len(episode_history) - 1, current_idx + 10)
                            ts_relative_idx = current_idx - back_idx
                            clean_coords = [frame['coords'] for frame in episode_history]

                            save_dir = os.path.join(molecule_dir, f"ts_structures/zone_{zone_id}")
                            os.makedirs(save_dir, exist_ok=True)
                            save_dynamic_trajectory(
                                clean_coords[back_idx : forward_idx + 1],
                                save_dir, env.mol.elem, ts_relative_idx,
                                current_grad, current_raw_grad
                            )

                            if zone_id not in zone_records:
                                zone_records[zone_id] = {}

                            zone_records[zone_id]['grad']      = current_grad
                            zone_records[zone_id]['raw_grad']  = current_raw_grad
                            zone_records[zone_id]['spawn']     = spawn_coords
                            zone_records[zone_id]['ts_coords'] = episode_history[current_idx]['coords'].copy()
                            zone_records[zone_id]['target']    = episode_history[forward_idx]['coords'].copy()
                            zone_records[zone_id]['id']        = zone_id

                if info.get('absolute_convergence', False):
                    print("\n" + "*"*70)
                    print(f"   🎉 SUCCESS: {mol_name.upper()} (Phase 0 — Absolute Convergence)!")
                    print("*"*70)
                    save_dir = os.path.join(molecule_dir, "RL_Discovered_TS_FINAL")
                    clean_coords = [frame['coords'] for frame in episode_history]
                    save_dynamic_trajectory(clean_coords, save_dir, env.mol.elem, len(clean_coords)-1)
                    return True

                if len(memory) > 256: agent.update_parameters(memory, 256)

            current_best_tracked_grad = min([z['raw_grad'] for z in zone_records.values()]) if zone_records else float('inf')

            if current_best_tracked_grad < global_best_phase0_grad:
                global_best_phase0_grad = current_best_tracked_grad
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1

            print(f"Phase 0 | Ep {episode+1} | Patience: {episodes_without_improvement}/{PHASE_0_FALLBACK} | "
                  f"Tracking {len(zone_records)}/2 zones | Best Raw Grad: {global_best_phase0_grad:.5f}")

            if current_best_tracked_grad < 0.01:
                print(f"\n   >>> PHASE 0 GATE PASSED (raw_grad: {current_best_tracked_grad:.5f} < 0.01). Advancing to Phase 1.")
                break

            if episodes_without_improvement >= PHASE_0_FALLBACK:
                print("\n   -> Phase 0 Patience Limit Reached. Transitioning to Phase 1.")
                break

        # ==========================================
        # STAGE 2: SORT THE TOP ZONES
        # ==========================================
        valid_zones = list(zone_records.values())

        if not valid_zones:
            print(f"\n[!] No Transition States discovered in Phase 0 for {mol_name}. Skipping.")
            return False

        valid_zones.sort(key=lambda x: x['grad'])

        print("\n" + "="*60)
        print(f"   STAGE 2: SORTING VALID ZONES ({len(valid_zones)} FOUND)")
        for i, z in enumerate(valid_zones):
            print(f"      Rank {i+1}: Zone {z['id']} (Initial RMS: {z['grad']:.5f})")
        print("="*60)

        # ==========================================
        # STAGE 3: PHASE 1 — TIGHT REFINEMENT
        # ==========================================
        PHASE_1_FALLBACK = 20

        print("\n" + "="*60)
        print(f"   STAGE 3: PHASE 1 REFINEMENT (BREADTH-FIRST ACROSS ALL ZONES)")
        print("="*60)

        for rank, zone_data in enumerate(valid_zones):
            zone_num = zone_data['id']
            print(f"\n   >>> COMMENCING PHASE 1 FOR ZONE {zone_num} (Rank {rank+1})")

            current_phase = 1
            episodes_without_improvement = 0
            zone_best_grad     = zone_data['grad']
            zone_best_raw_grad = zone_data.get('raw_grad', 999.0)

            best_spawn  = zone_data['spawn'].copy()
            env.set_custom_endpoints(best_spawn, p_zmat.coords.flatten(),
                                     mode_text=f"Phase 1 Zone {zone_num} (n_neg=0 spawn → global product)")

            current_save_dir = os.path.join(molecule_dir, f"render_phase_1/zone_{zone_num}")
            os.makedirs(current_save_dir, exist_ok=True)
            ts_dir = os.path.join(current_save_dir, "best_ts")

            for episode in range(250):
                phase1_episodes += 1  # Track Episode
                improved_this_episode = False
                state = env.reset()
                done = False

                episode_history = [{'coords': env.current_coords.copy(), 'n_neg': getattr(env, 'last_n_neg', 0)}]

                while not done:
                    phase1_total_steps += 1  # Track Steps
                    action = agent.select_action(state)
                    next_state, reward, done, info = env.step(action, current_phase)

                    n_val = info.get('n_neg', 0)
                    episode_history.append({'coords': env.current_coords.copy(), 'n_neg': n_val})
                    memory.push(state, action, reward, next_state, 1 if done else 0)
                    state = next_state

                    if info.get('is_ts', False):
                        current_grad     = info.get('grad_rms', 999.0)
                        current_raw_grad = info.get('raw_grad', 999.0)

                        if current_grad < molecule_best_grad:
                            molecule_best_grad = current_grad
                            molecule_best_raw_grad = current_raw_grad
                            molecule_best_traj = [frame['coords'] for frame in episode_history]
                            molecule_best_zone = str(zone_num)

                        if current_grad < zone_best_grad:
                            print(f"   [!] Zone {zone_num} Improved! RMS: {current_grad:.5f} (Prev: {zone_best_grad:.5f})")
                            zone_best_grad     = current_grad
                            zone_best_raw_grad = current_raw_grad
                            improved_this_episode = True

                            current_idx = len(episode_history) - 1

                            back_idx = current_idx
                            while back_idx > 0 and episode_history[back_idx]['n_neg'] != 0:
                                back_idx -= 1
                            if back_idx == 0 and episode_history[0]['n_neg'] != 0:
                                spawn_coords = best_spawn
                            else:
                                spawn_coords = episode_history[back_idx]['coords'].copy()

                            forward_idx = min(len(episode_history) - 1, current_idx + 15)
                            ts_relative_idx = current_idx - back_idx
                            clean_coords = [frame['coords'] for frame in episode_history]
                            save_dynamic_trajectory(
                                clean_coords[back_idx : forward_idx + 1],
                                ts_dir, env.mol.elem, ts_relative_idx,
                                zone_best_grad, zone_best_raw_grad
                            )

                            best_spawn = spawn_coords
                            env.set_custom_endpoints(best_spawn, p_zmat.coords.flatten(),
                                                     mode_text=f"Phase 1 Sliding Spawn Zone {zone_num}")

                    # --- CONVERGENCE CHECK: raw gradient norm ---
                    if info.get('is_ts', False) and info.get('raw_grad', 999.0) < 0.005:
                        print("\n" + "*"*70)
                        print(f"   🎉 SUCCESS: {mol_name.upper()} (Phase 1 — Raw Gradient Converged)!")
                        print(f"   Raw Gradient Norm: {info['raw_grad']:.6f} < 0.005")
                        print("*"*70)
                        save_dir = os.path.join(molecule_dir, "RL_Discovered_TS_FINAL")
                        clean_coords = [frame['coords'] for frame in episode_history]
                        save_dynamic_trajectory(clean_coords, save_dir, env.mol.elem, len(clean_coords)-1)
                        return True

                    if info.get('absolute_convergence', False):
                        print("\n" + "*"*70)
                        print(f"   🎉 SUCCESS: {mol_name.upper()} (Phase 1 — Absolute Convergence)!")
                        print("*"*70)
                        save_dir = os.path.join(molecule_dir, "RL_Discovered_TS_FINAL")
                        clean_coords = [frame['coords'] for frame in episode_history]
                        save_dynamic_trajectory(clean_coords, save_dir, env.mol.elem, len(clean_coords)-1)
                        return True

                    if len(memory) > 256: agent.update_parameters(memory, 256)

                if improved_this_episode:
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1

                print(f"Zone {zone_num} | Phase 1 | Ep {episode+1} | "
                      f"Patience: {episodes_without_improvement}/{PHASE_1_FALLBACK} | "
                      f"Best Grad: {zone_best_grad:.5f} | Best Raw Grad: {zone_best_raw_grad:.5f}")

                if episodes_without_improvement >= PHASE_1_FALLBACK:
                    print(f"\n   >>> ZONE {zone_num}: PHASE 1 COMPLETE — patience exhausted (Best Raw Grad: {zone_best_raw_grad:.5f}).")
                    break

        # ==========================================
        # BEST EFFORT FALLBACK
        # ==========================================
        print(f"\n   [!] Agent exhausted all zones for {mol_name} without success.")

        if len(molecule_best_traj) > 0:
            print("\n" + "="*70)
            print(f"   ⚠️  SAVING BEST EFFORT: {mol_name.upper()}")
            print(f"   Best RMS Gradient: {molecule_best_grad:.6f} | Best Raw Gradient: {molecule_best_raw_grad:.6f} | Zone {molecule_best_zone}")
            print("="*70)
            save_dir = os.path.join(molecule_dir, "RL_Discovered_TS_BEST_EFFORT")
            save_dynamic_trajectory(molecule_best_traj, save_dir, env.mol.elem, len(molecule_best_traj)-1)

        return False

    finally:
        # ==========================================
        # SAVE RUN DETAILS (Guaranteed to execute)
        # ==========================================
        details_path = os.path.join(molecule_dir, "run_details.txt")
        
        # Calculate averages safely
        p0_avg_steps = (phase0_total_steps / phase0_episodes) if phase0_episodes > 0 else 0
        p1_avg_steps = (phase1_total_steps / phase1_episodes) if phase1_episodes > 0 else 0
        
        with open(details_path, "w") as f:
            f.write(f"=== Agent Run Details: {mol_name} ===\n")
            f.write(f"Phase 0 Total Episodes: {phase0_episodes}\n")
            f.write(f"Phase 0 Average Steps/Episode: {p0_avg_steps:.2f}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Phase 1 Total Episodes: {phase1_episodes}\n")
            f.write(f"Phase 1 Average Steps/Episode: {p1_avg_steps:.2f}\n")
            
        print(f"   [+] Saved agent metrics to {details_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None,
                        help="Folder of reactions (default: ../data relative to this script)")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir is not None:
        DATA_DIR = os.path.abspath(args.data_dir)
    else:
        DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

    if not os.path.exists(DATA_DIR):
        print(f"FATAL: Could not locate data directory at {DATA_DIR}")
        return

    molecule_folders = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]
    molecule_folders.sort()

    print(f"Found {len(molecule_folders)} reactions in {DATA_DIR}")
    print(f"Mode: INDEPENDENT EVALUATION (fresh agent per reaction, no transfer)\n")

    dummy_feature_size = 8 * 5

    n_success, n_best_effort, n_skipped = 0, 0, 0

    for folder in molecule_folders:
        mol_name = os.path.basename(folder)

        # ==========================================
        # RESUME LOGIC: skip already-evaluated reactions
        # ==========================================
        final_ts_dir = os.path.join(folder, "RL_Discovered_TS_FINAL")
        best_effort_dir = os.path.join(folder, "RL_Discovered_TS_BEST_EFFORT")

        if os.path.exists(final_ts_dir):
            print(f"   [Resume] Skipping {mol_name.upper()}: already SUCCEEDED in a previous run.")
            n_skipped += 1
            continue
        if os.path.exists(best_effort_dir):
            print(f"   [Resume] Skipping {mol_name.upper()}: best-effort already saved in a previous run.")
            n_skipped += 1
            continue

        # ==========================================
        # FRESH agent and memory for THIS reaction only
        # ==========================================
        print(f"\n   [Setup] Initializing FRESH agent for {mol_name}...")
        agent = SACAgent(dummy_feature_size, 3)
        memory = ReplayBuffer(600000)

        success = train_molecule(agent, memory, folder, mol_name)

        if success:
            n_success += 1
        else:
            n_best_effort += 1

        # ==========================================
        # CLEANUP
        # ==========================================
        print(f"   [Cleanup] Discarding agent + memory for {mol_name}...")
        del agent, memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("   === EVALUATION COMPLETE ===")
    print(f"   Successes (raw gradient < 0.005): {n_success}")
    print(f"   Best-effort fallbacks: {n_best_effort}")
    print(f"   Skipped (already done): {n_skipped}")
    print("="*60)


if __name__ == "__main__":
    main()