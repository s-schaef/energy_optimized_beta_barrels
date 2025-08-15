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
    """
    Suppress cleanup errors in worker processes by clearing atexit handlers
    and redirecting stderr during process termination.
    """
    # Clear all atexit handlers to prevent cleanup errors
    atexit._clear()
    
    # Register a minimal cleanup function that suppresses stderr
    def silent_exit():
        # Redirect stderr to devnull during exit to suppress any remaining errors
        try:
            sys.stderr = open(os.devnull, 'w')
        except:
            pass
    
    atexit.register(silent_exit)

def evaluate_single_geometry(params_and_monomer_and_subunits):
    """
    Worker function to evaluate a single geometry in parallel.
    Each process creates its own RingBuilder to avoid sharing issues.
    
    Parameters:
        params_and_monomer_and_subunits (tuple): (parameters_dict, monomer_pdb_path, n_subunits)
        
    Returns:
        dict: Evaluation results
    """
    # Suppress cleanup errors in worker process
    suppress_worker_cleanup()
    
    params, monomer_pdb, n_subunits = params_and_monomer_and_subunits  # Unpack n_subunits
    
    # Suppress ALL warnings including DeprecationWarnings
    warnings.filterwarnings('ignore')
    
    # Redirect stdout and stderr to devnull to suppress output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull
    
    try:
        # Create RingBuilder instance for this process (with suppressed output)
        builder = RingBuilder(monomer_pdb)
        
        # Use a temporary directory for this evaluation to ensure cleanup
        with tempfile.TemporaryDirectory(prefix='ring_eval_') as temp_dir:            
            # Build ring with specified output location
            builder.build_ring(
                n_subunits=n_subunits,  # Use the passed n_subunits
                radius=params['radius'],
                tilt_angle=params['tilt_angle'],
            )       
            # Score ring
            print('scoring ring with parameters:', params)
            scores = builder.score_ring()
                        
            # Add geometry parameters to results
            scores.update(params)
        
        return scores
        
    except Exception as e:
        print(f"Error evaluating parameters {params}: {e}")
        # Return failed result with high energy
        failed_result = params.copy()
        failed_result.update({
            'total_score': 999999,
            'fa_atr': 999999,
            'fa_rep': 999999,
            'hbond_sr_bb': 0,
            'hbond_lr_bb': 0,
            'error': str(e)
        })
        return failed_result
        
    finally:
        # Restore stdout and stderr
        try:
            devnull.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass

class RingOptimizer:
    """
    Optimize ring geometry using parallel coarse-to-fine grid search.
    """
    
    def __init__(self, monomer_pdb: str, n_subunits: int, gasdermin: bool = False, n_processes: int = None):
        """
        Initialize optimizer with aligned monomer.
        
        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
            n_processes (int): Number of processes to use. If None, uses all available cores.
        """
        self.monomer_pdb = os.path.abspath(monomer_pdb)  # Use absolute path for multiprocessing
        self.n_subunits = n_subunits
        self.gasdermin = gasdermin  # Flag for gasdermin-specific modifications
        # Set up multiprocessing
        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = min(n_processes, mp.cpu_count())
        
        print(f"Using {self.n_processes} processes for parallel optimization")
        
        # Create a single builder instance for main process calculations
        self.builder = RingBuilder(self.monomer_pdb, self.gasdermin)  # gasdermin flag can be set here if needed
        
        # Calculate adaptive separation distance range
        self.base_radius = self._calculate_base_radius()
        print(f"Estimated radius around: {self.base_radius:.2f} Ã…")

        # Base tilt angle for the subunit
        self.base_tilt_angle = 0.0  # Default tilt angle in degrees
        print(f"Using base tilt angle: {self.base_tilt_angle}Â°")

    def _calculate_base_radius(self) -> float:
        """
        Calculate the base radius for the ring assembly based on the number of monomers.
        
        Returns:
            float: Base radius in Angstroms
        """
        monomer_atoms = self.builder.monomer_atoms.select_atoms('backbone')

        # get y-coordinates of all backbone atoms
        y_coords = monomer_atoms.positions[:, 1]
        # get min and max y-coordinates
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)

        # calculate the base width as the difference between max and min y-coordinates
        monomer_width = max_y - min_y 

        # calculate circumference of the ring
        circumference = monomer_width * self.n_subunits

        # calculate radius from circumference
        radius = circumference / (2 * np.pi)

        return radius
    
    def _generate_parameter_grid(self, 
                                 radius_range: Tuple[float, float],
                                 tilt_angle_range: Tuple[float, float],
                                 number_configurations: int) -> List[Dict]:
        """
        Generate parameter combinations for grid search.
        
        Parameters:
            radius_range (tuple): (min, max) radius values
            tilt_angle_range (tuple): (min, max) tilt angle values

        Returns:
            List[Dict]: List of parameter combinations
        """
        # Generate radius points
        radius_min, radius_max = radius_range
        radii = np.linspace(radius_min, radius_max, number_configurations)

        # Generate tilt angle points
        tilt_angle_min, tilt_angle_max = tilt_angle_range
        tilt_angles = np.linspace(tilt_angle_min, tilt_angle_max, number_configurations)    

        # Create parameter combinations
        radius_grid, tilt_grid = np.meshgrid(radii, tilt_angles)
        parameter_combinations = [
            {'radius': r, 'tilt_angle': t} 
            for r, t in zip(radius_grid.flatten(), tilt_grid.flatten())
        ]

        return parameter_combinations
    
    def _evaluate_parameter_set_parallel(self, parameter_combinations: List[Dict]) -> pd.DataFrame:
        """
        Evaluate a set of parameter combinations in parallel.
        
        Parameters:
            parameter_combinations (list): List of parameter dictionaries
            
        Returns:
            pd.DataFrame: Results with scores and parameters
        """
        total_combinations = len(parameter_combinations)
        print(f"Evaluating {total_combinations} parameter combinations using {self.n_processes} processes...")
        
        # Prepare input for worker processes
        work_items = [(params, self.monomer_pdb, self.n_subunits) for params in parameter_combinations]
        
        start_time = time.time()
        results = []
        
        # Try to use spawn method for cleaner process separation, but fall back gracefully
        original_start_method = mp.get_start_method()
        use_spawn = False
        
        try:
            # Check if spawn is available and try to use it
            available_methods = mp.get_all_start_methods()
            if 'spawn' in available_methods and original_start_method != 'spawn':
                try:
                    mp.set_start_method('spawn', force=True)
                    use_spawn = True
                    print(f"Using 'spawn' method for multiprocessing (original: {original_start_method})")
                except Exception as e:
                    print(f"Could not set spawn method: {e}, using {original_start_method}")
            else:
                print(f"Using '{original_start_method}' method for multiprocessing")
            
            with mp.Pool(processes=self.n_processes) as pool:
                # Use imap for progress tracking
                result_iter = pool.imap(evaluate_single_geometry, work_items)
                
                for i, result in enumerate(result_iter):
                    results.append(result)
                    if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
                        progress = (i + 1) / total_combinations * 100
                        elapsed = time.time() - start_time
                        eta = elapsed * (total_combinations - i - 1) / (i + 1) if i > 0 else 0
                        print(f"Progress: {i+1}/{total_combinations} ({progress:.1f}%) - ETA: {eta/60:.1f} minutes")
                
                # Properly close and join the pool
                pool.close()
                pool.join()
                
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            raise
        except Exception as e:
            print(f"Error in parallel evaluation: {e}")
            print("Falling back to sequential evaluation...")
            return self._evaluate_parameter_set_sequential(parameter_combinations)
        finally:
            # Restore original start method if we changed it
            if use_spawn:
                try:
                    mp.set_start_method(original_start_method, force=True)
                    print(f"Restored multiprocessing method to '{original_start_method}'")
                except Exception as e:
                    print(f"Warning: Could not restore original start method: {e}")
        
        elapsed = time.time() - start_time
        print(f"Completed {total_combinations} evaluations in {elapsed/60:.1f} minutes")
        
        return pd.DataFrame(results)
    
    def _evaluate_parameter_set_sequential(self, parameter_combinations: List[Dict]) -> pd.DataFrame:
        """
        Fallback sequential evaluation if parallel fails.
        
        Parameters:
            parameter_combinations (list): List of parameter dictionaries
            
        Returns:
            pd.DataFrame: Results with scores and parameters
        """
        print("Running sequential evaluation...")
        results = []
        total_combinations = len(parameter_combinations)
        start_time = time.time()
        
        for i, params in enumerate(parameter_combinations):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time = elapsed / i
                    remaining = (total_combinations - i) * avg_time
                    print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%) - "
                          f"ETA: {remaining/60:.1f} minutes")
            
            try:
                # Build ring with current parameters
                self.builder.build_ring(
                    n_subunits=self.n_subunits,
                    radius=params['radius'],
                    tilt_angle=params['tilt_angle'],
                )
                
                scores = self.builder.score_ring()
                scores.update(params)
                results.append(scores)

            except Exception as e:
                print(f"Error evaluating parameters {params}: {e}")
                # Return failed result with high energy
                failed_result = params.copy()
                failed_result.update({
                    'total_score': 999999,
                    'fa_atr': 999999,
                    'fa_rep': 999999,
                    'hbond_sr_bb': 0,
                    'hbond_lr_bb': 0,
                    'error': str(e)
                })
                results.append(failed_result)
        
        elapsed = time.time() - start_time
        print(f"Completed {total_combinations} evaluations in {elapsed/60:.1f} minutes")
        
        return pd.DataFrame(results)
    
    def optimize(self,
                 optimization_rounds: int = 2,
                 radius_range: Tuple[float, float] = (80, 100),
                 angle_range: Tuple[float, float] = (-30, 30),
                 save_csv: bool = True) -> Dict:

        """
        Run complete coarse-to-fine optimization with asymmetric rotations.
        
        Parameters:
            subunit_range (tuple): (min, max) number of subunits to optimize for
            save_results (bool): Whether to save results to CSV files
            
        Returns:
            Dict: Optimization results with best parameters
        """
        print("="*70)
        
        number_configurations = 10  # Number of configurations per parameter range
        total_combinations = (number_configurations ** 2 * optimization_rounds)
        print(f"Total combinations: {total_combinations} in {optimization_rounds} rounds")

        if angle_range is not None:
            tilt_angle_range = angle_range
        else:
            # Default tilt angle range if not provided
            tilt_angle_range = (self.base_tilt_angle - 30, self.base_tilt_angle + 30)

        # Coarse search
        for round_num in range(optimization_rounds):
            print(f"\nRound {round_num + 1}/{optimization_rounds}")
            print(f"Will search radius range: {radius_range[0]:.2f} to {radius_range[1]:.2f}A")
            print(f"Will search tilt angle range: {tilt_angle_range[0]:.2f} to {tilt_angle_range[1]:.2f}°")

            radius_stepsize = (radius_range[1] - radius_range[0]) / number_configurations 
            tilt_angle_stepsize = (tilt_angle_range[1] - tilt_angle_range[0]) / number_configurations
            
            # Generate parameter combinations
            parameter_combinations = self._generate_parameter_grid(radius_range,
                                                               tilt_angle_range,
                                                               number_configurations)
            
            # Evaluate all combinations in parallel
            results = self._evaluate_parameter_set_parallel(parameter_combinations)
            # Sort by total score (lower is better)
            results = results.sort_values('total_score').reset_index(drop=True)
        
            if save_csv:
                results.to_csv(f'round{round_num}_results.csv', index=False)
                print(f"Results saved to 'round{round_num}_results.csv'")
                print("-"*70)

            # Get best results and update search parameters
            best_result = results.iloc[0]  # Best result is the first row after sorting

            self.base_radius = results.iloc[0:2]['radius'].mean() # take average of best two results
            self.base_tilt_angle = results.iloc[0:2]['tilt_angle'].mean() # take average of best two results

            # define new search ranges based on best result
            radius_range = (self.base_radius - radius_stepsize, self.base_radius + radius_stepsize)
            tilt_angle_range = (self.base_tilt_angle - tilt_angle_stepsize, self.base_tilt_angle + tilt_angle_stepsize)

        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Best parameters found:")
        print(f"  Radius: {best_result['radius']:.2f}A")
        print(f"  Tilt angle: {best_result['tilt_angle']:.2f}")
        print(f"  Total score: {best_result['total_score']:.2f}")
        print(f"  fa_atr: {best_result['fa_atr']:.2f} (reweighted *0.9, to allow minor overlap)")
        print(f"  fa_rep: {best_result['fa_rep']:.2f} (reweighted *0.02, to allow minor overlap)")
        print(f"  hbond_sr_bb: {best_result['hbond_sr_bb']:.2f}")
        print(f"  hbond_lr_bb: {best_result['hbond_lr_bb']:.2f} (reweighted *10, for better hydrogen bond optimization)")

        # write final structure to pdb file
        output_pdb = f"optimized_ring_{self.base_radius:.2f}A_{self.base_tilt_angle:.2f}deg.pdb"
        self.builder.build_ring(
            n_subunits=self.n_subunits,
            radius=self.base_radius,
            tilt_angle=self.base_tilt_angle,)
        self.builder.write_ring_pdb(output_pdb, centered=True)  
        
        return best_result.to_dict()

def main():
    """Main function for command-line usage."""    
    parser = argparse.ArgumentParser(description='Optimize dimer geometry for circular assembly using parallel asymmetric rotations')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--n_subunits', type=int, required=True, help='Number of subunits in the ring')
    parser.add_argument('--angle_range', type=float, nargs=2, default=[-30, 30],
                        help='Range of tilt angles for the beta-sheet in degrees (default: -30 to 30)')
    parser.add_argument('--radius_range', type=float, nargs=2, default=None,
                        help='Range of radii for the ring assembly in Angstroms (default: None)')
    parser.add_argument('--processes', type=int, help='Number of processes to use (default: all available cores)')
    parser.add_argument('--no_csv', action='store_true', help='Do not save results to CSV files')
    parser.add_argument('--rounds', type=int, default=2, help='Number of optimization rounds (default: 2)')
    parser.add_argument('--gasdermin', action='store_true', help='Use empirically tweaked values useful for gasdermin-family proteins (default: False)')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = RingOptimizer(args.monomer, args.n_subunits, args.gasdermin, n_processes=args.processes)
    if args.radius_range is None:
        # Use adaptive radius based on the monomer
        radius_range = (optimizer.base_radius * 0.6, optimizer.base_radius) # base radius most likely overestimates the radius, so we use 60% of it as minimum
    else:
        radius_range = tuple(args.radius_range)

    results = optimizer.optimize(
        optimization_rounds=args.rounds,  # Number of optimization rounds
        radius_range=tuple(radius_range),  # Convert to tuple for consistency
        angle_range=tuple(args.angle_range),  # Convert to tuple for consistency
        save_csv=not args.no_csv  # Save results to CSV unless --no_csv is specified
    )
    
    return results

if __name__ == '__main__':
    main()