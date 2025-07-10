#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import itertools
from dimer_builder import DimerBuilder
import time

class DimerOptimizer:
    """
    Optimize dimer geometry using coarse-to-fine grid search.
    """
    
    def __init__(self, monomer_pdb: str):
        """
        Initialize optimizer with aligned monomer.
        
        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
        """
        self.builder = DimerBuilder(monomer_pdb)
        self.monomer_pdb = monomer_pdb
        
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
        Generate parameter combinations for grid search.
        
        Parameters:
            separation_range (tuple): (min, max) separation distance
            separation_points (int): Number of separation distance points
            z_rotation_points (list): Z-rotation values to test
            x_rotation_points (list): X-rotation values to test
            y_rotation_points (list): Y-rotation values to test
            
        Returns:
            List[Dict]: List of parameter combinations
        """
        # Generate separation distance points
        sep_min, sep_max = separation_range
        separation_distances = np.linspace(sep_min, sep_max, separation_points)
        
        # Generate all combinations
        parameter_combinations = []
        for sep_dist, z_rot, x_rot, y_rot in itertools.product(
            separation_distances, z_rotation_points, x_rotation_points, y_rotation_points
        ):
            parameter_combinations.append({
                'separation_distance': sep_dist,
                'z_rotation': z_rot,
                'x_rotation': x_rot,
                'y_rotation': y_rot
            })
        
        return parameter_combinations
    
    def _evaluate_parameter_set(self, parameter_combinations: List[Dict]) -> pd.DataFrame:
        """
        Evaluate a set of parameter combinations.
        
        Parameters:
            parameter_combinations (list): List of parameter dictionaries
            
        Returns:
            pd.DataFrame: Results with scores and parameters
        """
        results = []
        total_combinations = len(parameter_combinations)
        
        print(f"Evaluating {total_combinations} parameter combinations...")
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
        separation_range = (self.base_separation * 0.8, self.base_separation * 1.2)
        separation_points = 5
        z_rotation_points = [0, 90, 180, 270]
        x_rotation_points = [-30, 0, 30]
        y_rotation_points = [-30, 0, 30]
        
        print(f"Coarse search parameters:")
        print(f"  Separation distance: {separation_points} points from {separation_range[0]:.1f} to {separation_range[1]:.1f} Å")
        print(f"  Z-rotation: {z_rotation_points}")
        print(f"  X-rotation: {x_rotation_points}")
        print(f"  Y-rotation: {y_rotation_points}")
        
        total_combinations = separation_points * len(z_rotation_points) * len(x_rotation_points) * len(y_rotation_points)
        print(f"Total combinations: {total_combinations}")
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_grid(
            separation_range, separation_points,
            z_rotation_points, x_rotation_points, y_rotation_points
        )
        
        # Evaluate all combinations
        results = self._evaluate_parameter_set(parameter_combinations)
        
        # Sort by total score (lower is better)
        results = results.sort_values('total_score').reset_index(drop=True)
        
        return results
    
    def fine_search(self, coarse_results: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
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
                  f"Z: {row['z_rotation']:.0f}, "
                  f"X: {row['x_rotation']:.0f}, "
                  f"Y: {row['y_rotation']:.0f}")
        
        all_fine_results = []
        
        for i, row in top_results.iterrows():
            print(f"\nRefining around result {i+1}...")
            
            # Define fine search ranges around this result
            sep_center = row['separation_distance']
            z_center = row['z_rotation']
            x_center = row['x_rotation'] 
            y_center = row['y_rotation']
            
            # Smaller ranges for fine search
            sep_range = (sep_center - 1.0, sep_center + 1.0)  # ±1 Å
            z_range = [z_center - 30, z_center - 15, z_center, z_center + 15, z_center + 30]  # ±30° in 15° steps
            x_range = [x_center - 15, x_center - 7.5, x_center, x_center + 7.5, x_center + 15]  # ±15° in 7.5° steps
            y_range = [y_center - 15, y_center - 7.5, y_center, y_center + 7.5, y_center + 15]  # ±15° in 7.5° steps
            
            # Handle angle wrapping for z_rotation
            z_range = [(z % 360) for z in z_range]
            
            parameter_combinations = self._generate_parameter_grid(
                sep_range, 5,  # 5 separation points
                z_range, x_range, y_range
            )
            
            fine_results = self._evaluate_parameter_set(parameter_combinations)
            all_fine_results.append(fine_results)
        
        # Combine all fine search results
        combined_fine_results = pd.concat(all_fine_results, ignore_index=True)
        
        # Remove duplicates and sort
        combined_fine_results = combined_fine_results.drop_duplicates(
            subset=['separation_distance', 'z_rotation', 'x_rotation', 'y_rotation']
        ).sort_values('total_score').reset_index(drop=True)
        
        return combined_fine_results
    
    def optimize(self, save_results: bool = True) -> Dict:
        """
        Run complete coarse-to-fine optimization.
        
        Parameters:
            save_results (bool): Whether to save results to CSV files
            
        Returns:
            Dict: Optimization results with best parameters
        """
        print("="*60)
        print("DIMER GEOMETRY OPTIMIZATION")
        print("="*60)
        
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
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Best parameters found:")
        print(f"  Separation distance: {best_result['separation_distance']:.2f} Å")
        print(f"  Z-rotation: {best_result['z_rotation']:.1f}°")
        print(f"  X-rotation: {best_result['x_rotation']:.1f}°") 
        print(f"  Y-rotation: {best_result['y_rotation']:.1f}°")
        print(f"  Total score: {best_result['total_score']:.2f}")
        print(f"  fa_atr: {best_result['fa_atr']:.2f}")
        print(f"  fa_rep: {best_result['fa_rep']:.2f}")
        print(f"  hbond_bb_sc: {best_result['hbond_bb_sc']:.2f}")
        
        # Calculate ring parameters for different sizes
        print(f"\nRing parameters for different assembly sizes:")
        for n_subunits in [50, 51, 52, 53, 54]:
            ring_params = self.builder.get_ring_geometry(
                best_result['separation_distance'],
                n_subunits,
                best_result['z_rotation'],
                best_result['x_rotation'], 
                best_result['y_rotation']
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
    
    parser = argparse.ArgumentParser(description='Optimize dimer geometry for circular assembly')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--no_save', action='store_true', help='Do not save results to CSV files')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = DimerOptimizer(args.monomer)
    results = optimizer.optimize(save_results=not args.no_save)
    
    return results


if __name__ == "__main__":
    main()