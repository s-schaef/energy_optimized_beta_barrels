#!/usr/bin/env python3
"""
Example workflow demonstrating how to build and optimize a beta-barrel assembly.

This script shows the three-step process:
  1. Align a monomer to standard orientation
  2. Optimize ring geometry with pore-quality-focused scoring
  3. Build the final ring with the best parameters

Usage:
    python examples/example_workflow.py --input monomer.pdb --n_subunits 24
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment_module import align_monomer_from_file
from optimization_module import RingOptimizer
from ring_builder import RingBuilder


def run_workflow(input_pdb: str, n_subunits: int, rounds: int = 3,
                 gasdermin: bool = False, processes: int = None):
    """Run the full alignment -> optimization -> build workflow."""

    # Step 1: Align monomer
    print("=" * 70)
    print("STEP 1: Aligning monomer")
    print("=" * 70)
    aligned_pdb = input_pdb.replace('.pdb', '_aligned.pdb')
    aligner = align_monomer_from_file(input_pdb, aligned_pdb)
    print(f"Aligned monomer saved to {aligned_pdb}\n")

    # Step 2: Optimize ring geometry
    print("=" * 70)
    print("STEP 2: Optimizing ring geometry")
    print("=" * 70)
    optimizer = RingOptimizer(
        aligned_pdb,
        n_subunits=n_subunits,
        gasdermin=gasdermin,
        n_processes=processes,
    )

    best = optimizer.optimize(
        optimization_rounds=rounds,
        rank_by='pore_quality',  # Prioritize pore quality over raw energy
    )

    print(f"\nBest geometry found:")
    print(f"  Radius:     {best['radius']:.2f} A")
    print(f"  Tilt angle: {best['tilt_angle']:.2f} deg")
    print(f"  Z-offset:   {best.get('z_offset', 0):.2f} A")
    print(f"  Pore quality:    {best['pore_quality']:.2f}")
    print(f"  H-bonds/subunit: {best.get('hbond_per_subunit', 0):.2f}")

    # Step 3: Build final ring with best parameters
    print("\n" + "=" * 70)
    print("STEP 3: Building final ring")
    print("=" * 70)
    final_pdb = f"final_ring_{n_subunits}mer.pdb"
    builder = RingBuilder(aligned_pdb, gasdermin=gasdermin)
    builder.build_ring(
        n_subunits=n_subunits,
        radius=best['radius'],
        tilt_angle=best['tilt_angle'],
        z_offset=best.get('z_offset', 0.0),
    )
    builder.write_ring_pdb(final_pdb)
    print(f"\nFinal ring saved to {final_pdb}")

    # Print scoring summary
    scores = builder.score_ring()
    print(f"\nFinal ring scores:")
    for key, val in scores.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full beta-barrel assembly workflow')
    parser.add_argument('--input', required=True, help='Input monomer PDB file')
    parser.add_argument('--n_subunits', type=int, required=True,
                        help='Number of subunits in the ring')
    parser.add_argument('--rounds', type=int, default=3,
                        help='Optimization rounds (default: 3)')
    parser.add_argument('--gasdermin', action='store_true',
                        help='Enable gasdermin-specific modifications')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes')

    args = parser.parse_args()
    run_workflow(args.input, args.n_subunits, args.rounds,
                 args.gasdermin, args.processes)
