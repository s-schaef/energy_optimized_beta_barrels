#!/usr/bin/env python3
"""
Python API version of the command-line workflow:

  1. Align a single protomer to the standard orientation
  2. Screen ring geometries (radius, tilt angle) for a fixed number of subunits;
     the best screened geometry is written to optimized_ring_*.pdb
  3. Optionally build and score a ring with parameters chosen after inspecting
     the screen (e.g. against a cryo-EM density)

Usage (after `pip install -e .`), e.g. with the protomer extracted by
run_6vfe_gsdmd.sh:

    python examples/example_workflow.py --input examples/output/6vfe/gsdmd_protomer.pdb \\
        --n_subunits 33 --cone_angle 10
    python examples/example_workflow.py --input examples/output/6vfe/gsdmd_protomer.pdb \\
        --n_subunits 33 --cone_angle 10 --skip_screen --radius 128 --tilt_angle -20
"""

import os
import argparse
from barrel_builder import align_monomer_from_file, RingOptimizer, RingBuilder


def run_workflow(input_pdb: str, n_subunits: int, rounds: int = 2,
                 cone_angle: float = 0.0, processes: int = None,
                 skip_screen: bool = False, radius: float = None,
                 tilt_angle: float = None):
    """Run the alignment -> screen -> build workflow."""

    print("=" * 70)
    print("STEP 1: Aligning monomer")
    print("=" * 70)
    aligned_pdb = f"{os.path.splitext(input_pdb)[0]}_aligned.pdb"
    align_monomer_from_file(input_pdb, aligned_pdb)

    best = None
    if not skip_screen:
        print("\n" + "=" * 70)
        print("STEP 2: Screening ring geometries")
        print("=" * 70)
        optimizer = RingOptimizer(
            aligned_pdb,
            n_subunits=n_subunits,
            cone_angle=cone_angle,
            n_processes=processes,
        )
        best = optimizer.optimize(optimization_rounds=rounds)

    if radius is not None and tilt_angle is not None:
        print("\n" + "=" * 70)
        print("STEP 3: Building ring with chosen parameters")
        print("=" * 70)
        final_pdb = f"ring_{n_subunits}mer_{radius:.1f}A_{tilt_angle:.1f}deg.pdb"
        builder = RingBuilder(aligned_pdb)
        builder.output_pdb = final_pdb
        builder.build_ring(n_subunits=n_subunits, radius=radius, tilt_angle=tilt_angle,
                           cone_angle=cone_angle)
        scores = builder.score_ring()
        print(f"Total score: {scores['total_score']:.2f}")
        print(f"Ring written to {final_pdb}")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align -> screen -> build workflow')
    parser.add_argument('--input', required=True, help='Single-protomer PDB file')
    parser.add_argument('--n_subunits', type=int, required=True,
                        help='Number of subunits in the ring')
    parser.add_argument('--rounds', type=int, default=2,
                        help='Screening rounds (default: 2)')
    parser.add_argument('--cone_angle', type=float, default=0.0,
                        help='Rotation around the tangential (y) axis in degrees (default: 0.0)')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes')
    parser.add_argument('--skip_screen', action='store_true',
                        help='Skip the screen (e.g. when the geometry is already known)')
    parser.add_argument('--radius', type=float, help='Radius for the final ring (A)')
    parser.add_argument('--tilt_angle', type=float, help='Tilt angle for the final ring (deg)')

    args = parser.parse_args()
    run_workflow(args.input, args.n_subunits, args.rounds, args.cone_angle,
                 args.processes, args.skip_screen, args.radius, args.tilt_angle)
