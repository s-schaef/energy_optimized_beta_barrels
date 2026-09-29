#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='Bio')
warnings.filterwarnings('ignore', message='.*Bio.Application.*')

import os
import sys
import time
import atexit
import argparse
import importlib.util
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Dict, List, Tuple
from barrel_builder.ring_builder import RingBuilder, SCORE_WEIGHTS

PYROSETTA_HINT = (
    "PyRosetta is required for scoring but could not be imported. Install it into "
    "the active environment, e.g.:\n"
    "    pip install pyrosetta-installer\n"
    "    python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'"
)


def suppress_worker_cleanup():
    """Suppress cleanup errors in worker processes."""
    atexit._clear()

    def silent_exit():
        try:
            sys.stderr = open(os.devnull, 'w')
        except:
            pass

    atexit.register(silent_exit)


def failed_result(params: Dict, error: Exception) -> Dict:
    """Result row for a geometry that could not be scored (sorted last)."""
    result = params.copy()
    result.update({term: np.nan for term in SCORE_WEIGHTS})
    result.update({'total_score': np.inf, 'error': str(error)})
    return result


def evaluate_single_geometry(args_tuple):
    """
    Worker function to evaluate a single geometry in parallel.
    Each task creates its own RingBuilder; PyRosetta is initialized once per process.
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
        )
        scores = builder.score_ring()
        scores.update(params)
        return scores

    except Exception as e:
        return failed_result(params, e)

    finally:
        try:
            devnull.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass


class RingOptimizer:
    """
    Screen ring geometries with a parallel coarse-to-fine grid search.

    Explores radius and tilt_angle for a fixed number of subunits and ranks
    the rigid-body assemblies by the reweighted PyRosetta total_score. This is
    a coarse screen: no minimization is performed and the scores are not
    physical energies.
    """

    def __init__(self, monomer_pdb: str, n_subunits: int,
                 gasdermin: bool = False, n_processes: int = None):
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

    def _calculate_base_radius(self) -> float:
        """Radius at which the monomers' y-extents would just touch around the ring."""
        backbone = self.builder.monomer_atoms.select_atoms('backbone')
        y_coords = backbone.positions[:, 1]
        monomer_width = np.max(y_coords) - np.min(y_coords)
        circumference = monomer_width * self.n_subunits
        return circumference / (2 * np.pi)

    def _generate_parameter_grid(self,
                                 radius_range: Tuple[float, float],
                                 tilt_angle_range: Tuple[float, float],
                                 grid_size: int) -> List[Dict]:
        radii = np.unique(np.linspace(radius_range[0], radius_range[1], grid_size))
        tilt_angles = np.unique(np.linspace(tilt_angle_range[0], tilt_angle_range[1], grid_size))

        r_grid, t_grid = np.meshgrid(radii, tilt_angles)
        return [
            {'radius': r, 'tilt_angle': t}
            for r, t in zip(r_grid.flatten(), t_grid.flatten())
        ]

    def _evaluate_parallel(self, parameter_combinations: List[Dict]) -> pd.DataFrame:
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
                # Let workers exit normally; leaving the with-block would
                # terminate them (SIGTERM), which prints spurious tracebacks.
                pool.close()
                pool.join()

        except Exception as e:
            print(f"Parallel evaluation failed: {e}")
            print("Falling back to sequential evaluation...")
            return self._evaluate_sequential(parameter_combinations)

        elapsed = time.time() - start_time
        print(f"Completed {total} evaluations in {elapsed/60:.1f} minutes")
        return pd.DataFrame(results)

    def _evaluate_sequential(self, parameter_combinations: List[Dict]) -> pd.DataFrame:
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
                )
                scores = self.builder.score_ring()
                scores.update(params)
                results.append(scores)
            except Exception as e:
                print(f"Error evaluating {params}: {e}")
                results.append(failed_result(params, e))

        elapsed = time.time() - start_time
        print(f"Completed {total} evaluations in {elapsed/60:.1f} minutes")
        return pd.DataFrame(results)

    @staticmethod
    def _check_failures(results: pd.DataFrame):
        """Abort if every evaluation failed; warn if some did."""
        if 'error' not in results.columns:
            return
        failed = results['error'].notna()
        if failed.all():
            raise RuntimeError(
                f"All {len(results)} evaluations failed. First error: "
                f"{results.loc[failed, 'error'].iloc[0]}")
        if failed.any():
            print(f"WARNING: {failed.sum()} of {len(results)} evaluations failed "
                  f"(see 'error' column). First error: {results.loc[failed, 'error'].iloc[0]}")

    def optimize(self,
                 optimization_rounds: int = 2,
                 radius_range: Tuple[float, float] = None,
                 angle_range: Tuple[float, float] = (-30, 30),
                 grid_size: int = 10,
                 save_csv: bool = True) -> Dict:
        """
        Run the coarse-to-fine grid search over radius and tilt angle.

        Each round evaluates a grid_size x grid_size grid. The next round is
        centered on the mean of the two best geometries, with the search range
        narrowed to +/- one grid step. The best evaluated geometry is written
        to optimized_ring_<radius>A_<tilt>deg.pdb.

        Parameters:
            optimization_rounds (int): Number of grid refinement rounds
            radius_range (tuple): (min, max) radius in Angstroms; defaults to
                60-100% of the estimated base radius
            angle_range (tuple): (min, max) tilt angle in degrees
            grid_size (int): Grid points per dimension per round
            save_csv (bool): Write round{N}_results.csv for every round

        Returns:
            dict: Parameters and scores of the best evaluated geometry
        """
        if importlib.util.find_spec('pyrosetta') is None:
            raise ImportError(PYROSETTA_HINT)

        if radius_range is None:
            radius_range = (self.base_radius * 0.6, self.base_radius)

        tilt_angle_range = angle_range

        total_combinations = grid_size ** 2 * optimization_rounds
        print("=" * 70)
        print(f"Screening: {total_combinations} total evaluations over "
              f"{optimization_rounds} rounds")
        print("=" * 70)

        best_result = None

        for round_num in range(optimization_rounds):
            print(f"\n--- Round {round_num + 1}/{optimization_rounds} ---")
            print(f"  Radius: {radius_range[0]:.2f} to {radius_range[1]:.2f} A")
            print(f"  Tilt:   {tilt_angle_range[0]:.2f} to {tilt_angle_range[1]:.2f} deg")

            radius_step = (radius_range[1] - radius_range[0]) / grid_size
            tilt_step = (tilt_angle_range[1] - tilt_angle_range[0]) / grid_size

            parameter_combinations = self._generate_parameter_grid(
                radius_range, tilt_angle_range, grid_size
            )

            results = self._evaluate_parallel(parameter_combinations)
            self._check_failures(results)
            results = results.sort_values('total_score').reset_index(drop=True)

            if save_csv:
                results.to_csv(f'round{round_num}_results.csv', index=False)
                print(f"Results saved to round{round_num}_results.csv")

            if best_result is None or results.iloc[0]['total_score'] < best_result['total_score']:
                best_result = results.iloc[0]

            print("\n  Top 3 results (ranked by total_score):")
            for i in range(min(3, len(results))):
                row = results.iloc[i]
                print(f"    #{i+1}: r={row['radius']:.2f} A, tilt={row['tilt_angle']:.2f} deg, "
                      f"total={row['total_score']:.1f}, "
                      f"hbond_lr_bb={row['hbond_lr_bb']:.1f}")

            # Center the next round on the mean of the two best geometries
            self.base_radius = results.iloc[0:2]['radius'].mean()
            self.base_tilt_angle = results.iloc[0:2]['tilt_angle'].mean()

            radius_range = (self.base_radius - radius_step, self.base_radius + radius_step)
            tilt_angle_range = (self.base_tilt_angle - tilt_step, self.base_tilt_angle + tilt_step)

        print("\n" + "=" * 70)
        print("SCREEN COMPLETE")
        print("=" * 70)
        print("Best evaluated geometry:")
        print(f"  Radius:     {best_result['radius']:.2f} A")
        print(f"  Tilt angle: {best_result['tilt_angle']:.2f} deg")
        print(f"  Total score: {best_result['total_score']:.2f}")
        for term, weight in SCORE_WEIGHTS.items():
            print(f"  {term}: {best_result[term]:.2f} (weight {weight})")

        output_pdb = (f"optimized_ring_{best_result['radius']:.2f}A_"
                      f"{best_result['tilt_angle']:.2f}deg.pdb")
        self.builder.build_ring(
            n_subunits=self.n_subunits,
            radius=best_result['radius'],
            tilt_angle=best_result['tilt_angle'],
        )
        self.builder.write_ring_pdb(output_pdb, centered=True)
        print("Inspect this model manually (e.g. against a cryo-EM density) before use.")

        return best_result.to_dict()


def main():
    """CLI entry point for barrel-optimize."""
    parser = argparse.ArgumentParser(
        description='Screen ring geometries (radius, tilt angle) for a circular '
                    'beta-barrel assembly of a fixed number of subunits')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--n_subunits', type=int, required=True,
                        help='Number of subunits in the ring (2-52)')
    parser.add_argument('--angle_range', type=float, nargs=2, default=[-30, 30],
                        help='Range of tilt angles in degrees (default: -30 30)')
    parser.add_argument('--radius_range', type=float, nargs=2, default=None,
                        help='Range of radii in Angstroms (default: 60-100%% of the '
                             'estimated base radius)')
    parser.add_argument('--grid_size', type=int, default=10,
                        help='Grid points per dimension per round (default: 10)')
    parser.add_argument('--rounds', type=int, default=2,
                        help='Number of refinement rounds (default: 2)')
    parser.add_argument('--processes', type=int,
                        help='Number of processes (default: all cores)')
    parser.add_argument('--no_csv', action='store_true',
                        help='Do not save results to CSV files')
    parser.add_argument('--gasdermin', action='store_true',
                        help='Apply the additional 10 degree y-rotation used for gasdermins')

    args = parser.parse_args()

    optimizer = RingOptimizer(
        args.monomer, args.n_subunits, args.gasdermin, n_processes=args.processes
    )

    try:
        optimizer.optimize(
            optimization_rounds=args.rounds,
            radius_range=tuple(args.radius_range) if args.radius_range else None,
            angle_range=tuple(args.angle_range),
            grid_size=args.grid_size,
            save_csv=not args.no_csv,
        )
    except (ImportError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
