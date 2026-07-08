#!/usr/bin/env python3
"""
Reorder the atoms of one XYZ file to match a reference, superpose them, and
report the RMSD before and after reordering.

The reordered + superposed structure is written to a new XYZ file expressed in
the *reference* coordinate frame, so you can load the reference and the output
together in any viewer (VMD, Avogadro, ASE gui, PyMOL...) and they should
overlay if the mapping is correct.

Usage:
    python align_rmsd.py reference.xyz target.xyz
    python align_rmsd.py reference.xyz target.xyz -o aligned.xyz -m inertia-hungarian

"target" is the file whose atoms get reordered onto "reference".
"""

import argparse
import sys
from collections import Counter

import numpy as np
import rmsd

REORDER = {
    "inertia-hungarian": rmsd.reorder_inertia_hungarian,
    "hungarian": rmsd.reorder_hungarian,
    "distance": rmsd.reorder_distance,
    "brute": rmsd.reorder_brute,
}


def load(path):
    """Return (symbols, atomic_numbers, coords) for an xyz file."""
    symbols, coords = rmsd.get_coordinates_xyz(path, return_atoms_as_int=False)
    znums, _ = rmsd.get_coordinates_xyz(path, return_atoms_as_int=True)
    return np.asarray(symbols), np.asarray(znums), np.asarray(coords)


def check_composition(za, zb):
    """Both structures must contain the same multiset of elements."""
    ca, cb = Counter(za.tolist()), Counter(zb.tolist())
    if ca != cb:
        sa = ", ".join(f"{rmsd.str_atom(int(z))}:{n}" for z, n in sorted(ca.items()))
        sb = ", ".join(f"{rmsd.str_atom(int(z))}:{n}" for z, n in sorted(cb.items()))
        sys.exit(
            "ERROR: the two files do not have the same atoms.\n"
            f"  reference: {sa}\n"
            f"  target   : {sb}\n"
            "A one-to-one mapping does not exist (different molecule, conformer "
            "with added/removed atoms, or different protonation state)."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="reference .xyz (atom order is kept)")
    ap.add_argument("target", help=".xyz whose atoms get reordered onto the reference")
    ap.add_argument("-o", "--output", default=None,
                    help="output file (default: <target>_aligned.xyz)")
    ap.add_argument("-m", "--method", default="inertia-hungarian",
                    choices=list(REORDER),
                    help="reordering method (default: inertia-hungarian)")
    args = ap.parse_args()

    out = args.output or args.target.rsplit(".", 1)[0] + "_aligned.xyz"

    sa, za, A = load(args.reference)        # reference: symbols, Z, coords
    sb, zb, B = load(args.target)           # target

    if len(za) != len(zb):
        sys.exit(f"ERROR: atom counts differ ({len(za)} vs {len(zb)}).")
    check_composition(za, zb)

    # --- RMSD with the original ordering (Kabsch superposition only) ---
    rmsd_before = rmsd.kabsch_rmsd(A, B, translate=True)

    # --- find the permutation that maps target (B) onto reference (A) ---
    # The reorder functions work on centered coordinates.
    Ac = A - rmsd.centroid(A)
    Bc = B - rmsd.centroid(B)
    view = REORDER[args.method](za, zb, Ac, Bc)   # index array: B[view] matches A

    zb_r = zb[view]
    sb_r = sb[view]
    Br = B[view]

    # sanity: after reordering, element identities must line up
    mismatch = int(np.sum(za != zb_r))
    if mismatch:
        print(f"WARNING: {mismatch} atom(s) map element-to-element incorrectly. "
              f"The '{args.method}' heuristic likely failed for this case; try a "
              f"different -m method or use spyrmsd (connectivity-based).",
              file=sys.stderr)

    # --- RMSD after reordering ---
    rmsd_after = rmsd.kabsch_rmsd(A, Br, translate=True)

    # --- superpose reordered target onto reference, in the reference frame ---
    cA = rmsd.centroid(A)
    Br_c = Br - rmsd.centroid(Br)
    Br_rot = rmsd.kabsch_rotate(Br_c, A - cA)   # rotate onto centered reference
    Br_out = Br_rot + cA                        # shift back into reference frame

    title = f"reordered+aligned onto {args.reference} (method={args.method})"
    with open(out, "w") as fh:
        block = rmsd.set_coordinates(sb_r, Br_out, title=title)
        fh.write(block if block.endswith("\n") else block + "\n")

    # --- report ---
    moved = int(np.sum(view != np.arange(len(view))))
    print("=" * 60)
    print(f"reference : {args.reference}  ({len(za)} atoms)")
    print(f"target    : {args.target}")
    print(f"method    : {args.method}")
    print("-" * 60)
    print(f"RMSD before reordering : {rmsd_before:10.6f}  (Kabsch only)")
    print(f"RMSD after  reordering : {rmsd_after:10.6f}  (reorder + Kabsch)")
    print(f"atoms whose index changed: {moved} / {len(view)}")
    print("-" * 60)
    print(f"wrote: {out}")
    print("Load the reference and this file together to confirm the overlay.")
    print("=" * 60)


if __name__ == "__main__":
    main()