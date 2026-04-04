#!/usr/bin/env python3

import warnings
# Suppress Biopython deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='Bio')
warnings.filterwarnings('ignore', message='.*Bio.Application.*')

import os
import sys
import time
import atexit
import tempfile
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Dict, List, Tuple
from ring_builder import RingBuilder


def suppress_worker_cleanup():
    """Suppress cleanup errors in worker processes."""
    atexit._clear()

    def silent_exit():
        try:
            sys.stderr = open(os.devnull, 'w')
        except:
            pass

    atexit.register(silent_exit)


def evaluate_single_geometry(args_tuple):
    """
    Worker function to evaluate a single geometry in parallel.
    Each process creates its own RingBuilder to avoid sharing issues.

    Parameters:
        args_tuple (tuple): (parameters_dict, monomer_pdb_path, n_subunits, gasdermin)

    Returns:
        dict: Evaluation results including pore_quality score
    """
    suppress_worker_cleanup()

    params, monomer_pdb, n_subunits, gasdermin = args_tuple

    warnings.filterwarnings('ignore')

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        builder = RingBuilder(monomer_pdb, gasdermin=gasdermin)

        builder.build_ring(
            n_subunits=n_subunits,
            radius=params['radius'],
            tilt_angle=params['tilt_angle'],
            z_offset=params.get('z_offset', 0.0),
        )
        scores = builder.score_ring()
        scores.update(params)

        return scores

    except Exception as e:
        failed_result = params.copy()
        failed_result.update({
            'total_score': 999999,
            'fa_atr': 999999,
            'fa_rep': 999999,
            'hbond_sr_bb': 0,
            'hbond_lr_bb': 0,
            'hbond_per_subunit': 0,
            'pore_quality': 999999,
            'error': str(e)
        })
        return failed_result

    finally:
        try:
            devnull.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass


class RingOptimizer:
    """
    Optimize ring geometry using parallel coarse-to-fine grid search.

    The optimizer explores three parameters:
      - radius: distance from ring center to subunit center
      - tilt_angle: beta-sheet rotation around x-axis
      - z_offset: alternating vertical stagger between adjacent subunits

    Results are ranked by pore_quality (inter-subunit hydrogen bonding)
    rather than raw total energy, ensuring the central beta-sheet pore
    is well-formed.
    """

    def __init__(self, monomer_pdb: str, n_subunits: int,
                 gasdermin: bool = False, n_processes: int = None):
        """
        Initialize optimizer with aligned monomer.

        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
            n_subunits (int): Number of subunits in the ring
            gasdermin (bool): Enable gasdermin-specific modifications
            n_processes (int): Number of processes (default: all cores)
        """
        self.monomer_pdb = os.path.abspath(monomer_pdb)
        self.n_subunits = n_subunits
        self.gasdermin = gasdermin

        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = min(n_processes, mp.cpu_count())

        print(f"Using {self.n_processes} processes for parallel optimization")

        self.builder = RingBuilder(self.monomer_pdb, gasdermin=self.gasdermin)

        self.base_radius = self._calculate_base_radius()
        print(f"Estimated base radius: {self.base_radius:.2f} A")

        self.base_tilt_angle = 0.0
        self.base_z_offset = 0.0
        print(f"Using base tilt angle: {self.base_tilt_angle} deg")
        print(f"Using base z_offset: {self.base_z_offset} A")

    def _calculate_base_radius(self) -> float:
        """
        Calculate base radius from monomer dimensions and subunit count.

        Uses the y-extent of backbone atoms to estimate the width each
        subunit occupies in the ring circumference.
        """
        backbone = self.builder.monomer_atoms.select_atoms('backbone')
        y_coords = backbone.positions[:, 1]
        monomer_width = np.max(y_coords) - np.min(y_coords)
        circumference = monomer_width * self.n_subunits
        return circumference / (2 * np.pi)

    def _generate_parameter_grid(self,
                                 radius_range: Tuple[float, float],
                                 tilt_angle_range: Tuple[float, float],
                                 z_offset_range: Tuple[float, float],
                                 grid_size: int) -> List[Dict]:
        """
        Generate parameter combinations for 3D grid search.

        Parameters:
            radius_range: (min, max) radius values
            tilt_angle_range: (min, max) tilt angle values
            z_offset_range: (min, max) z-offset values
            grid_size: number of points per dimension

        Returns:
            List[Dict]: Parameter combinations
        """
        radii = np.linspace(radius_range[0], radius_range[1], grid_size)
        tilt_angles = np.linspace(tilt_angle_range[0], tilt_angle_range[1], grid_size)
        z_offsets = np.linspace(z_offset_range[0], z_offset_range[1], grid_size)

        r_grid, t_grid, z_grid = np.meshgrid(radii, tilt_angles, z_offsets)
        return [
            {'radius': r, 'tilt_angle': t, 'z_offset': z}
            for r, t, z in zip(r_grid.flatten(), t_grid.flatten(), z_grid.flatten())
        ]

    def _evaluate_parallel(self, parameter_combinations: List[Dict]) -> pd.DataFrame:
        """
        Evaluate parameter combinations in parallel.

        Returns:
            pd.DataFrame: Results sorted by pore_quality (ascending = better)
        """
        total = len(parameter_combinations)
        print(f"Evaluating {total} parameter combinations using {self.n_processes} processes...")

        work_items = [
            (params, self.monomer_pdb, self.n_subunits, self.gasdermin)
            for params in parameter_combinations
        ]

        start_time = time.time()
        results = []

        try:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=self.n_processes) as pool:
                result_iter = pool.imap(evaluate_single_geometry, work_items)

                for i, result in enumerate(result_iter):
                    results.append(result)
                    if (i + 1) % 10 == 0 or (i + 1) == total:
                        progress = (i + 1) / total * 100
                        elapsed = time.time() - start_time
                        eta = elapsed * (total - i - 1) / (i + 1) if i > 0 else 0
                        print(f"Progress: {i+1}/{total} ({progress:.1f}%) - "
                              f"ETA: {eta/60:.1f} min")

        except Exception as e:
            print(f"Parallel evaluation failed: {e}")
            print("Falling back to sequential evaluation...")
            return self._evaluate_sequential(parameter_combinations)

        elapsed = time.time() - start_time
        print(f"Completed {total} evaluations in {elapsed/60:.1f} minutes")

        return pd.DataFrame(results)

    def _evaluate_sequential(self, parameter_combinations: List[Dict]) -> pd.DataFrame:
        """Fallback sequential evaluation."""
        print("Running sequential evaluation...")
        results = []
        total = len(parameter_combinations)
        start_time = time.time()

        for i, params in enumerate(parameter_combinations):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (total - i) / (i + 1) if i > 0 else 0
                print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%) - "
                      f"ETA: {eta/60:.1f} min")

            try:
                self.builder.build_ring(
                    n_subunits=self.n_subunits,
                    radius=params['radius'],
                    tilt_angle=params['tilt_angle'],
                    z_offset=params.get('z_offset', 0.0),
                )
                scores = self.builder.score_ring()
                scores.update(params)
                results.append(scores)
            except Exception as e:
                print(f"Error evaluating {params}: {e}")
                failed_result = params.copy()
                failed_result.update({
                    'total_score': 999999,
                    'fa_atr': 999999,
                    'fa_rep': 999999,
                    'hbond_sr_bb': 0,
                    'hbond_lr_bb': 0,
                    'hbond_per_subunit': 0,
                    'pore_quality': 999999,
                    'error': str(e)
                })
                results.append(failed_result)

        elapsed = time.time() - start_time
        print(f"Completed {total} evaluations in {elapsed/60:.1f} minutes")
        return pd.DataFrame(results)

    def optimize(self,
                 optimization_rounds: int = 3,
                 radius_range: Tuple[float, float] = None,
                 angle_range: Tuple[float, float] = (-30, 30),
                 z_offset_range: Tuple[float, float] = (0, 5),
                 grid_size: int = 8,
                 rank_by: str = 'pore_quality',
                 save_csv: bool = True) -> Dict:
        """
        Run coarse-to-fine grid search optimization.

        The optimizer searches over radius, tilt angle, and z-offset in
        successive rounds. Each round narrows the search window around the
        best result from the previous round.

        Parameters:
            optimization_rounds (int): Number of refinement rounds (default: 3)
            radius_range (tuple): (min, max) radius. If None, derived from monomer.
            angle_range (tuple): (min, max) tilt angle in degrees
            z_offset_range (tuple): (min, max) z-offset in Angstroms
            grid_size (int): Points per dimension per round (default: 8)
            rank_by (str): Score column to rank by. 'pore_quality' focuses on
                inter-subunit H-bonds; 'total_score' uses raw energy.
            save_csv (bool): Save per-round results to CSV

        Returns:
            Dict: Best parameters and scores found
        """
        if radius_range is None:
            radius_range = (self.base_radius * 0.6, self.base_radius)

        tilt_angle_range = angle_range
        z_off_range = z_offset_range

        total_combinations = grid_size ** 3 * optimization_rounds
        print("=" * 70)
        print(f"Optimizing: {total_combinations} total evaluations over "
              f"{optimization_rounds} rounds")
        print(f"Ranking by: {rank_by}")
        print("=" * 70)

        best_result = None

        for round_num in range(optimization_rounds):
            print(f"\n--- Round {round_num + 1}/{optimization_rounds} ---")
            print(f"  Radius: {radius_range[0]:.2f} to {radius_range[1]:.2f} A")
            print(f"  Tilt:   {tilt_angle_range[0]:.2f} to {tilt_angle_range[1]:.2f} deg")
            print(f"  Z-off:  {z_off_range[0]:.2f} to {z_off_range[1]:.2f} A")

            radius_step = (radius_range[1] - radius_range[0]) / max(grid_size - 1, 1)
            tilt_step = (tilt_angle_range[1] - tilt_angle_range[0]) / max(grid_size - 1, 1)
            z_step = (z_off_range[1] - z_off_range[0]) / max(grid_size - 1, 1)

            parameter_combinations = self._generate_parameter_grid(
                radius_range, tilt_angle_range, z_off_range, grid_size
            )

            results = self._evaluate_parallel(parameter_combinations)

            # Rank by the chosen metric (lower = better for both)
            results = results.sort_values(rank_by).reset_index(drop=True)

            if save_csv:
                results.to_csv(f'round{round_num}_results.csv', index=False)
                print(f"Results saved to round{round_num}_results.csv")

            best_result = results.iloc[0]

            # Print top results for this round
            print(f"\n  Top 3 results (ranked by {rank_by}):")
            for i in range(min(3, len(results))):
                row = results.iloc[i]
                print(f"    #{i+1}: r={row['radius']:.2f} A, tilt={row['tilt_angle']:.2f} deg, "
                      f"z_off={row.get('z_offset', 0):.2f} A, "
                      f"pore_quality={row['pore_quality']:.1f}, "
                      f"hbond_lr_bb={row['hbond_lr_bb']:.1f}, "
                      f"total={row['total_score']:.1f}")

            # Narrow search around the best result for next round
            best_r = best_result['radius']
            best_t = best_result['tilt_angle']
            best_z = best_result.get('z_offset', 0.0)

            radius_range = (best_r - radius_step, best_r + radius_step)
            tilt_angle_range = (best_t - tilt_step, best_t + tilt_step)
            z_off_range = (max(0, best_z - z_step), best_z + z_step)

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Best parameters:")
        print(f"  Radius:     {best_result['radius']:.2f} A")
        print(f"  Tilt angle: {best_result['tilt_angle']:.2f} deg")
        print(f"  Z-offset:   {best_result.get('z_offset', 0):.2f} A")
        print(f"  Pore quality:  {best_result['pore_quality']:.2f}")
        print(f"  H-bond/subunit: {best_result.get('hbond_per_subunit', 0):.2f}")
        print(f"  Total score:   {best_result['total_score']:.2f}")
        print(f"  fa_atr:  {best_result['fa_atr']:.2f} (weight 0.9)")
        print(f"  fa_rep:  {best_result['fa_rep']:.2f} (weight 0.02)")
        print(f"  hbond_sr_bb: {best_result['hbond_sr_bb']:.2f} (weight 1.0)")
        print(f"  hbond_lr_bb: {best_result['hbond_lr_bb']:.2f} (weight 10.0)")

        # Write final structure using the actual best parameters
        output_pdb = (f"optimized_ring_{best_result['radius']:.2f}A_"
                      f"{best_result['tilt_angle']:.2f}deg_"
                      f"{best_result.get('z_offset', 0):.2f}z.pdb")
        self.builder.build_ring(
            n_subunits=self.n_subunits,
            radius=best_result['radius'],
            tilt_angle=best_result['tilt_angle'],
            z_offset=best_result.get('z_offset', 0.0),
        )
        self.builder.write_ring_pdb(output_pdb, centered=True)

        return best_result.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description='Optimize ring geometry for circular beta-barrel assembly')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--n_subunits', type=int, required=True,
                        help='Number of subunits in the ring')
    parser.add_argument('--angle_range', type=float, nargs=2, default=[-30, 30],
                        help='Range of tilt angles in degrees (default: -30 30)')
    parser.add_argument('--radius_range', type=float, nargs=2, default=None,
                        help='Range of radii in Angstroms (default: adaptive)')
    parser.add_argument('--z_offset_range', type=float, nargs=2, default=[0, 5],
                        help='Range of z-offsets in Angstroms (default: 0 5)')
    parser.add_argument('--grid_size', type=int, default=8,
                        help='Grid points per dimension per round (default: 8)')
    parser.add_argument('--processes', type=int,
                        help='Number of processes (default: all cores)')
    parser.add_argument('--no_csv', action='store_true',
                        help='Do not save results to CSV files')
    parser.add_argument('--rounds', type=int, default=3,
                        help='Number of optimization rounds (default: 3)')
    parser.add_argument('--rank_by', choices=['pore_quality', 'total_score'],
                        default='pore_quality',
                        help='Score to rank by (default: pore_quality)')
    parser.add_argument('--gasdermin', action='store_true',
                        help='Enable gasdermin-specific modifications')

    args = parser.parse_args()

    optimizer = RingOptimizer(
        args.monomer, args.n_subunits, args.gasdermin, n_processes=args.processes
    )

    if args.radius_range is None:
        radius_range = (optimizer.base_radius * 0.6, optimizer.base_radius)
    else:
        radius_range = tuple(args.radius_range)

    results = optimizer.optimize(
        optimization_rounds=args.rounds,
        radius_range=radius_range,
        angle_range=tuple(args.angle_range),
        z_offset_range=tuple(args.z_offset_range),
        grid_size=args.grid_size,
        rank_by=args.rank_by,
        save_csv=not args.no_csv,
    )

    return results


if __name__ == '__main__':
    main()
