#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import itertools
from dimer_builder import DimerBuilder
import time
import multiprocessing as mp
from functools import partial
import os
import tempfile
import shutil

def evaluate_single_geometry(params_and_monomer):
    """
    Worker function to evaluate a single geometry in parallel.
    Each process creates its own DimerBuilder to avoid sharing issues.
    
    Parameters:
        params_and_monomer (tuple): (parameters_dict, monomer_pdb_path)
        
    Returns:
        dict: Evaluation results
    """
    params, monomer_pdb = params_and_monomer
    
    try:
        # Create DimerBuilder instance for this process
        builder = DimerBuilder(monomer_pdb, initialize_pyrosetta=True)
        
        # Evaluate geometry
        scores = builder.evaluate_geometry(**params)
        return scores
        
    except Exception as e:
        # Return failed result with high energy
        failed_result = params.copy()
        failed_result.update({
            'total_score': 999999,
            'fa_atr': 999999,
            'fa_rep': 999999,
            'hbond_bb_sc': 0,
            'error': str(e)
        })
        return failed_result

class DimerOptimizer:
    """
    Optimize dimer geometry using parallel coarse-to-fine grid search with asymmetric rotations.
    """
    
    def __init__(self, monomer_pdb: str, n_processes: int = None):
        """
        Initialize optimizer with aligned monomer.
        
        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
            n_processes (int): Number of processes to use. If None, uses all available cores.
        """
        self.monomer_pdb = os.path.abspath(monomer_pdb)  # Use absolute path for multiprocessing
        
        # Set up multiprocessing
        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = min(n_processes, mp.cpu_count())
        
        print(f"Using {self.n_processes} processes for parallel optimization")
        
        # Create a single builder instance for main process calculations
        self.builder = DimerBuilder(self.monomer_pdb)
        
        # Calculate adaptive separation distance range
        self.base_separation = self._calculate_ca_footprint_radius()
        print(f"Calculated CA footprint radius: {self.base_separation:.2f} Å")
        print(f"Will search separation distances: {self.base_separation*0.8:.1f} - {self.base_separation*1.2:.1f} Å")
    
    def _calculate_ca_footprint_radius(self) -> float:
        """
        Calculate the xy-footprint radius of the monomer using CA atoms.
        
        Returns:
            float: Radius that encompasses the CA atoms in xy-plane
        """
        ca_atoms = self.builder.monomer_atoms.select_atoms('name CA')
        
        if len(ca_atoms) == 0:
            raise ValueError("No CA atoms found in monomer")
        
        # Get CA positions in xy-plane
        ca_positions = ca_atoms.positions[:, :2]  # Only x and y coordinates
        
        # Calculate center of mass in xy
        center_xy = ca_positions.mean(axis=0)
        
        # Calculate distances from center
        distances = np.linalg.norm(ca_positions - center_xy, axis=1)
        
        # Use 95th percentile to avoid outliers
        radius = np.percentile(distances, 95)
        
        return radius
    
    def _generate_parameter_grid(self, 
                                separation_range: Tuple[float, float],
                                separation_points: int,
                                z_rotation_points: List[float],
                                x_rotation_points: List[float], 
                                y_rotation_points: List[float]) -> List[Dict]:
        """
        Generate parameter combinations for grid search with asymmetric rotations.
        Z-rotations limited to 180° range since both subunits rotate independently.
        
        Parameters:
            separation_range (tuple): (min, max) separation distance
            separation_points (int): Number of separation distance points
            z_rotation_points (list): Z-rotation values to test (0-180° range)
            x_rotation_points (list): X-rotation values to test
            y_rotation_points (list): Y-rotation values to test
            
        Returns:
            List[Dict]: List of parameter combinations
        """
        # Generate separation distance points
        sep_min, sep_max = separation_range
        separation_distances = np.linspace(sep_min, sep_max, separation_points)
        
        parameter_combinations = []
        
        # Asymmetric search: each monomer can have different rotations
        # Z-rotations use reduced 180° range since relative orientations repeat after 180°
        for sep_dist in separation_distances:
            for z1, x1, y1, z2, x2, y2 in itertools.product(
                z_rotation_points, x_rotation_points, y_rotation_points,  # Monomer 1
                z_rotation_points, x_rotation_points, y_rotation_points   # Monomer 2
            ):
                parameter_combinations.append({
                    'separation_distance': sep_dist,
                    'z_rotation_1': z1,
                    'x_rotation_1': x1,
                    'y_rotation_1': y1,
                    'z_rotation_2': z2,
                    'x_rotation_2': x2,
                    'y_rotation_2': y2
                })
        
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
        work_items = [(params, self.monomer_pdb) for params in parameter_combinations]
        
        start_time = time.time()
        results = []
        
        # Use multiprocessing Pool for parallel evaluation
        try:
            with mp.Pool(processes=self.n_processes) as pool:
                # Use imap for progress tracking
                result_iter = pool.imap(evaluate_single_geometry, work_items)
                
                for i, result in enumerate(result_iter):
                    results.append(result)
                    
                    # Progress updates
                    if (i + 1) % 100 == 0 or (i + 1) == total_combinations:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (i + 1)
                        remaining = (total_combinations - i - 1) * avg_time
                        print(f"Progress: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%) - "
                              f"ETA: {remaining/60:.1f} minutes")
                
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            raise
        except Exception as e:
            print(f"Error in parallel evaluation: {e}")
            print("Falling back to sequential evaluation...")
            return self._evaluate_parameter_set_sequential(parameter_combinations)
        
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
            if i % 100 == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time = elapsed / i
                    remaining = (total_combinations - i) * avg_time
                    print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%) - "
                          f"ETA: {remaining/60:.1f} minutes")
            
            try:
                scores = self.builder.evaluate_geometry(**params)
                results.append(scores)
            except Exception as e:
                print(f"Error evaluating {params}: {e}")
                # Add failed result with high energy
                failed_result = params.copy()
                failed_result.update({
                    'total_score': 999999,
                    'fa_atr': 999999,
                    'fa_rep': 999999,
                    'hbond_bb_sc': 0
                })
                results.append(failed_result)
        
        elapsed = time.time() - start_time
        print(f"Completed {total_combinations} evaluations in {elapsed/60:.1f} minutes")
        
        return pd.DataFrame(results)
    
    def coarse_search(self) -> pd.DataFrame:
        """
        Perform coarse grid search over parameter space.
        
        Returns:
            pd.DataFrame: Results sorted by total score
        """
        print("Starting coarse grid search...")
        
        # Define coarse search parameters
        # Z-rotations: 0-180° range (6 points every 36°)
        # X,Y rotations: ±45° range (5 points every 22.5°)
        separation_range = (self.base_separation * 0.8, self.base_separation * 1.2)
        separation_points = 5
        z_rotation_points = [0, 36, 72, 108, 144, 180]  # 6 points in 180° range
        x_rotation_points = [-45, -22.5, 0, 22.5, 45]   # 5 points in ±45° range
        y_rotation_points = [-45, -22.5, 0, 22.5, 45]   # 5 points in ±45° range
        
        print(f"Coarse search parameters:")
        print(f"  Separation distance: {separation_points} points from {separation_range[0]:.1f} to {separation_range[1]:.1f} Å")
        print(f"  Z-rotation: {z_rotation_points} (180° range)")
        print(f"  X-rotation: {x_rotation_points}")
        print(f"  Y-rotation: {y_rotation_points}")
        
        total_combinations = (separation_points * 
                            len(z_rotation_points)**2 * 
                            len(x_rotation_points)**2 * 
                            len(y_rotation_points)**2)
        
        print(f"Total combinations: {total_combinations}")
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_grid(
            separation_range, separation_points,
            z_rotation_points, x_rotation_points, y_rotation_points
        )
        
        # Evaluate all combinations in parallel
        results = self._evaluate_parameter_set_parallel(parameter_combinations)
        
        # Sort by total score (lower is better)
        results = results.sort_values('total_score').reset_index(drop=True)
        
        return results
    
    def fine_search(self, coarse_results: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Perform fine grid search around the best results from coarse search.
        
        Parameters:
            coarse_results (pd.DataFrame): Results from coarse search
            top_n (int): Number of top results to refine around
            
        Returns:
            pd.DataFrame: Fine search results
        """
        print(f"\nStarting fine grid search around top {top_n} results...")
        
        # Get top results
        top_results = coarse_results.head(top_n)
        print("Top results from coarse search:")
        for i, row in top_results.iterrows():
            print(f"  {i+1}. Score: {row['total_score']:.2f}, "
                  f"Sep: {row['separation_distance']:.1f}, "
                  f"Z1: {row['z_rotation_1']:.0f}, Z2: {row['z_rotation_2']:.0f}, "
                  f"X1: {row['x_rotation_1']:.0f}, X2: {row['x_rotation_2']:.0f}")
        
        all_fine_combinations = []
        
        for i, row in top_results.iterrows():
            print(f"\nGenerating fine grid around result {i+1}...")
            
            # Define fine search ranges around this result
            sep_center = row['separation_distance']
            
            # Monomer 1 rotations
            z1_center = row['z_rotation_1']
            x1_center = row['x_rotation_1'] 
            y1_center = row['y_rotation_1']
            
            # Monomer 2 rotations
            z2_center = row['z_rotation_2']
            x2_center = row['x_rotation_2']
            y2_center = row['y_rotation_2']
            
            # Fine search ranges
            sep_range = (sep_center - 1.0, sep_center + 1.0)  # ±1.0 Å
            
            # ±18° in 6° steps for z-rotations (5 points each)
            z1_range = [z1_center - 18, z1_center - 9, z1_center, z1_center + 9, z1_center + 18]
            z2_range = [z2_center - 18, z2_center - 9, z2_center, z2_center + 9, z2_center + 18]
            
            # ±11.25° in 5.625° steps for x,y rotations (5 points each)  
            x1_range = [x1_center - 11.25, x1_center - 5.625, x1_center, x1_center + 5.625, x1_center + 11.25]
            y1_range = [y1_center - 11.25, y1_center - 5.625, y1_center, y1_center + 5.625, y1_center + 11.25]
            x2_range = [x2_center - 11.25, x2_center - 5.625, x2_center, x2_center + 5.625, x2_center + 11.25]
            y2_range = [y2_center - 11.25, y2_center - 5.625, y2_center, y2_center + 5.625, y2_center + 11.25]
            
            # Handle angle bounds
            z1_range = [max(0, min(180, z)) for z in z1_range]  # Keep z in 0-180° range
            z2_range = [max(0, min(180, z)) for z in z2_range]
            
            # Generate fine grid
            sep_distances = np.linspace(sep_range[0], sep_range[1], 5)  # 5 separation points
            
            for sep_dist in sep_distances:
                for z1, x1, y1, z2, x2, y2 in itertools.product(
                    z1_range, x1_range, y1_range, z2_range, x2_range, y2_range
                ):
                    all_fine_combinations.append({
                        'separation_distance': sep_dist,
                        'z_rotation_1': z1,
                        'x_rotation_1': x1,
                        'y_rotation_1': y1,
                        'z_rotation_2': z2,
                        'x_rotation_2': x2,
                        'y_rotation_2': y2
                    })
        
        print(f"\nTotal fine search combinations: {len(all_fine_combinations)}")
        
        # Remove duplicates
        df_combinations = pd.DataFrame(all_fine_combinations)
        parameter_columns = ['separation_distance', 'z_rotation_1', 'x_rotation_1', 'y_rotation_1',
                           'z_rotation_2', 'x_rotation_2', 'y_rotation_2']
        df_combinations = df_combinations.drop_duplicates(subset=parameter_columns)
        unique_combinations = df_combinations.to_dict('records')
        
        print(f"Unique combinations after deduplication: {len(unique_combinations)}")
        
        # Evaluate all fine combinations in parallel
        fine_results = self._evaluate_parameter_set_parallel(unique_combinations)
        
        # Sort by total score
        fine_results = fine_results.sort_values('total_score').reset_index(drop=True)
        
        return fine_results
    
    def optimize(self, subunit_range: Tuple[int, int] = (6, 16), 
                save_results: bool = True) -> Dict:
        """
        Run complete coarse-to-fine optimization with asymmetric rotations.
        
        Parameters:
            subunit_range (tuple): (min, max) number of subunits to optimize for
            save_results (bool): Whether to save results to CSV files
            
        Returns:
            Dict: Optimization results with best parameters
        """
        min_subunits, max_subunits = subunit_range
        print("="*70)
        print("PARALLEL DIMER GEOMETRY OPTIMIZATION - ASYMMETRIC ROTATIONS")
        print(f"Optimizing for {min_subunits}-{max_subunits} subunit rings")
        print(f"Using {self.n_processes} CPU cores")
        print("Z-rotations: 0-180° range (reduced due to symmetry)")
        print("X,Y rotations: ±45° range")
        print("="*70)
        
        # Coarse search
        coarse_results = self.coarse_search()
        
        if save_results:
            coarse_results.to_csv('coarse_search_results.csv', index=False)
            print(f"Coarse search results saved to 'coarse_search_results.csv'")
        
        # Fine search
        fine_results = self.fine_search(coarse_results)
        
        if save_results:
            fine_results.to_csv('fine_search_results.csv', index=False)
            print(f"Fine search results saved to 'fine_search_results.csv'")
        
        # Get best result
        best_result = fine_results.iloc[0]
        
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Best parameters found:")
        print(f"  Separation distance: {best_result['separation_distance']:.2f} Å")
        print(f"  Monomer 1 rotations: Z={best_result['z_rotation_1']:.1f}°, "
              f"X={best_result['x_rotation_1']:.1f}°, Y={best_result['y_rotation_1']:.1f}°")
        print(f"  Monomer 2 rotations: Z={best_result['z_rotation_2']:.1f}°, "
              f"X={best_result['x_rotation_2']:.1f}°, Y={best_result['y_rotation_2']:.1f}°")
        print(f"  Total score: {best_result['total_score']:.2f}")
        print(f"  fa_atr: {best_result['fa_atr']:.2f}")
        print(f"  fa_rep: {best_result['fa_rep']:.2f}")
        print(f"  hbond_bb_sc: {best_result['hbond_bb_sc']:.2f}")
        
        # Calculate ring parameters for the specified subunit range
        print(f"\nRing parameters for {min_subunits}-{max_subunits} subunit assemblies:")
        subunit_range_list = list(range(min_subunits, max_subunits + 1, max(1, (max_subunits - min_subunits) // 6)))
        if max_subunits not in subunit_range_list:
            subunit_range_list.append(max_subunits)
            
        for n_subunits in subunit_range_list:
            ring_params = self.builder.get_ring_geometry(
                best_result['separation_distance'],
                n_subunits,
                best_result['z_rotation_1'],
                best_result['x_rotation_1'], 
                best_result['y_rotation_1'],
                best_result['z_rotation_2'],
                best_result['x_rotation_2'],
                best_result['y_rotation_2']
            )
            print(f"  {n_subunits}-mer: radius = {ring_params['radius']:.1f} Å")
        
        return {
            'best_parameters': best_result.to_dict(),
            'coarse_results': coarse_results,
            'fine_results': fine_results
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize dimer geometry for circular assembly using parallel asymmetric rotations')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--subunit_range', type=int, nargs=2, default=[6, 16], 
                        help='Range of subunit numbers to optimize for (min max), default: 6 16')
    parser.add_argument('--processes', type=int, help='Number of processes to use (default: all available cores)')
    parser.add_argument('--no_save', action='store_true', help='Do not save results to CSV files')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = DimerOptimizer(args.monomer, n_processes=args.processes)
    results = optimizer.optimize(
        subunit_range=tuple(args.subunit_range), 
        save_results=not args.no_save
    )
    
    return results


if __name__ == "__main__":
    main()