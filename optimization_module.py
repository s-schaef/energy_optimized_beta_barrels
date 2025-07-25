#!/usr/bin/env python3

import warnings
# Suppress Biopython deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='Bio')
warnings.filterwarnings('ignore', message='.*Bio.Application.*')

import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Dict, List, Tuple
from ring_builder import RingBuilder

def evaluate_single_geometry(params_and_monomer_and_subunits):
    """
    Worker function to evaluate a single geometry in parallel.
    Each process creates its own RingBuilder to avoid sharing issues.
    
    Parameters:
        params_and_monomer_and_subunits (tuple): (parameters_dict, monomer_pdb_path, n_subunits)
        
    Returns:
        dict: Evaluation results
    """
    params, monomer_pdb, n_subunits = params_and_monomer_and_subunits  # Unpack n_subunits
    
    # Suppress all output from worker processes
    import sys
    import os
    import warnings
    import tempfile

    # Suppress ALL warnings including DeprecationWarnings
    warnings.filterwarnings('ignore')
    
    # Redirect stdout and stderr to devnull
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
    
    def __init__(self, monomer_pdb: str, n_subunits: int, angle_range: tuple = None, n_processes: int = None):
        """
        Initialize optimizer with aligned monomer.
        
        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
            n_processes (int): Number of processes to use. If None, uses all available cores.
        """
        self.monomer_pdb = os.path.abspath(monomer_pdb)  # Use absolute path for multiprocessing
        self.n_subunits = n_subunits
        # Set up multiprocessing
        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = min(n_processes, mp.cpu_count())
        
        print(f"Using {self.n_processes} processes for parallel optimization")
        
        # Create a single builder instance for main process calculations
        self.builder = RingBuilder(self.monomer_pdb)
        
        # Calculate adaptive separation distance range
        self.base_radius = self._calculate_base_radius()
        print(f"Estimated radius around: {self.base_radius:.2f} Å")

        # Base tilt angle for the subunit
        self.base_tilt_angle = 0.0  # Default tilt angle in degrees
        print(f"Using base tilt angle: {self.base_tilt_angle}°")

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
        # Use multiprocessing Pool for parallel evaluation
        try:
            with mp.Pool(processes=self.n_processes) as pool:
                # Use imap for progress tracking
                result_iter = pool.imap(evaluate_single_geometry, work_items)
                
                for i, result in enumerate(result_iter):
                    results.append(result)
                
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            raise
        except Exception as e:
            print(f"Error in parallel evaluation: {e}")
            print("Falling back to sequential evaluation...")
            return self._evaluate_parameter_set_sequential(parameter_combinations) #TODO: implement sequential fallback
        
        elapsed = time.time() - start_time
        print(f"Completed {total_combinations} evaluations in {elapsed/60:.1f} minutes")
        
        return pd.DataFrame(results)
    
    # def _evaluate_parameter_set_sequential(self, parameter_combinations: List[Dict]) -> pd.DataFrame:
    #     """
    #     Fallback sequential evaluation if parallel fails.
        
    #     Parameters:
    #         parameter_combinations (list): List of parameter dictionaries
            
    #     Returns:
    #         pd.DataFrame: Results with scores and parameters
    #     """
    #     print("Running sequential evaluation...")
    #     results = []
    #     total_combinations = len(parameter_combinations)
    #     start_time = time.time()
        
    #     for i, params in enumerate(parameter_combinations):
    #         if i % 100 == 0:
    #             elapsed = time.time() - start_time
    #             if i > 0:
    #                 avg_time = elapsed / i
    #                 remaining = (total_combinations - i) * avg_time
    #                 print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%) - "
    #                       f"ETA: {remaining/60:.1f} minutes")
            
    #         try:
    #             scores = self.builder.score_ring()
    #             results.append(scores)
    #         except Exception as e:
    #             print(f"Error evaluating {params}: {e}")
    #             # Add failed result with high energy
    #             failed_result = params.copy()
    #             failed_result.update({
    #                 'total_score': 999999,
    #                 'fa_atr': 999999,
    #                 'fa_rep': 999999,
    #                 'hbond_sr_bb': 0,
    #                 'hbond_lr_bb': 0,
    #             })
    #             results.append(failed_result)
        
    #     elapsed = time.time() - start_time
    #     print(f"Completed {total_combinations} evaluations in {elapsed/60:.1f} minutes")
        
    #     return pd.DataFrame(results)
    
    
    def optimize(self,
                 optimization_rounds: int = 2,
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
        # print("PARALLEL DIMER GEOMETRY OPTIMIZATION - ASYMMETRIC ROTATIONS")
        # print(f"Optimizing for {min_subunits}-{max_subunits} subunit rings")
        # print(f"Using {self.n_processes} CPU cores")
        # print("Z-rotations: 0-180° range (reduced due to symmetry)")
        # print("X,Y rotations: ±45° range")
        print("="*70)
        
        number_configurations = 10  # Number of configurations per parameter range
        total_combinations = (number_configurations ** 2 * optimization_rounds)
        print(f"Total combinations: {total_combinations} in {optimization_rounds} rounds")

        radius_range = (self.base_radius * 0.6, self.base_radius) # radius is most likely overestimated by base radius calculation
        tilt_angle_range = (self.base_tilt_angle - 30, self.base_tilt_angle + 30)


        # Coarse search
        for round in range(optimization_rounds):
            print(f"Will search radius range: {radius_range[0]:.2f} to {radius_range[1]:.2f} Å")
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
                results.to_csv(f'round{round}_results.csv', index=False)
                print(f"Results saved to 'round{round}_results.csv'")
        
            # Get best result and update search parameters
            best_result = results.iloc[0]
            self.base_radius = best_result['radius']
            self.base_tilt_angle = best_result['tilt_angle']

            # define new search ranges based on best result
            radius_range = (self.base_radius - radius_stepsize*2, self.base_radius + radius_stepsize*2)
            tilt_angle_range = (self.base_tilt_angle - tilt_angle_stepsize*2, self.base_tilt_angle + tilt_angle_stepsize*2)



        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Best parameters found:")
        print(f"  Radius: {best_result['radius']:.2f} Å")
        print(f"  Tilt angle: {best_result['tilt_angle']:.2f}°")
        print(f"  Total score: {best_result['total_score']:.2f}")
        print(f"  fa_atr: {best_result['fa_atr']:.2f}")
        print(f"  fa_rep: {best_result['fa_rep']:.2f}")
        print(f"  hbond_sr_bb: {best_result['hbond_sr_bb']:.2f}")
        print(f"  hbond_lr_bb: {best_result['hbond_lr_bb']:.2f}")

        
            
        # write final structure to pdb file
        output_pdb = f"optimized_ring_{self.base_radius:.2f}A_{self.base_tilt_angle:.2f}deg.pdb"
        self.builder.build_ring(
            n_subunits=self.n_subunits,
            radius=self.base_radius,
            tilt_angle=self.base_tilt_angle,)
        self.builder.write_ring_pdb(output_pdb, centered=True)  

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize dimer geometry for circular assembly using parallel asymmetric rotations')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--n_subunits', type=int, required=True, help='Number of subunits in the ring')
    parser.add_argument('--angle_range', type=float, nargs=2, default=[-30, 30],
                        help='Range of tilt angles for the beta-sheet in degrees (default: -30 to 30)')
    parser.add_argument('--processes', type=int, help='Number of processes to use (default: all available cores)')
    parser.add_argument('--no_csv', action='store_true', help='Do not save results to CSV files')
    parser.add_argument('--rounds', type=int, default=2, help='Number of optimization rounds (default: 2)')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = RingOptimizer(args.monomer, args.n_subunits, args.angle_range, n_processes=args.processes)
    results = optimizer.optimize(
        optimization_rounds=args.rounds,  # Number of optimization rounds
        save_csv=not args.no_csv  # Save results to CSV unless --no_csv is specified
    )
    
    return results


if __name__ == "__main__":
    main()