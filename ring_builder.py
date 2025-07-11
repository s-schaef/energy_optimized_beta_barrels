#!/usr/bin/env python3

import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import rotate
from string import ascii_uppercase as auc
from string import ascii_lowercase as alc
from typing import Dict, Optional
import shutil
import tempfile

class OptimizedRingBuilder:
    """
    Build circular protein assemblies using energy-optimized geometry parameters.
    """
    
    def __init__(self, monomer_pdb: str):
        """
        Initialize ring builder with aligned monomer.
        
        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
        """
        self.monomer_pdb = monomer_pdb
        self.monomer_universe = mda.Universe(monomer_pdb)
        self.monomer_atoms = self.monomer_universe.select_atoms('protein')
        
        # Create segment ID list (A-Z, then a-z for up to 52 subunits)
        self.segid_list = list(auc + alc)
        
    def calculate_ring_radius(self, separation_distance: float, n_subunits: int) -> float:
        """
        Calculate ring radius from dimer separation distance.
        
        Parameters:
            separation_distance (float): Optimal separation from dimer optimization
            n_subunits (int): Number of subunits in ring
            
        Returns:
            float: Ring radius in Angstroms
        """
        angle_between_subunits = 2 * np.pi / n_subunits
        radius = separation_distance / (2 * np.sin(angle_between_subunits / 2))
        return radius
    
    def build_ring_from_optimization(self, 
                                   optimization_results: Dict,
                                   n_subunits: int,
                                   output_pdb: str,
                                   create_intermediates: bool = False) -> str:
        """
        Build a ring using results from dimer optimization.
        
        Parameters:
            optimization_results (dict): Results from DimerOptimizer.optimize()
            n_subunits (int): Number of subunits in the ring
            output_pdb (str): Output PDB filename
            create_intermediates (bool): Whether to keep intermediate files
            
        Returns:
            str: Path to output PDB file
        """
        best_params = optimization_results['best_parameters']
        
        return self.build_ring(
            n_subunits=n_subunits,
            separation_distance=best_params['separation_distance'],
            z_rotation=best_params['z_rotation'],
            x_rotation=best_params['x_rotation'],
            y_rotation=best_params['y_rotation'],
            output_pdb=output_pdb,
            create_intermediates=create_intermediates
        )
    
    def build_ring(self,
                  n_subunits: int,
                  separation_distance: float,
                  z_rotation: float = 0.0,
                  x_rotation: float = 0.0,
                  y_rotation: float = 0.0,
                  output_pdb: str = None,
                  create_intermediates: bool = False) -> str:
        """
        Build a circular ring assembly with specified parameters.
        
        Parameters:
            n_subunits (int): Number of subunits in the ring
            separation_distance (float): Separation distance from dimer optimization
            z_rotation (float): Z-axis rotation (degrees)
            x_rotation (float): X-axis rotation (degrees)
            y_rotation (float): Y-axis rotation (degrees)
            output_pdb (str): Output PDB filename
            create_intermediates (bool): Whether to save intermediate subunit files
            
        Returns:
            str: Path to output PDB file
        """
        if n_subunits > len(self.segid_list):
            raise ValueError(f"Too many subunits ({n_subunits}). Maximum supported: {len(self.segid_list)}")
        
        # Calculate ring radius
        radius = self.calculate_ring_radius(separation_distance, n_subunits)
        
        print(f"Building {n_subunits}-mer ring:")
        print(f"  Separation distance: {separation_distance:.2f} Å")
        print(f"  Ring radius: {radius:.2f} Å") 
        print(f"  Rotations: Z={z_rotation:.1f}°, X={x_rotation:.1f}°, Y={y_rotation:.1f}°")
        
        # Create intermediates directory if requested
        intermediates_dir = None
        if create_intermediates:
            intermediates_dir = os.path.dirname(output_pdb) if output_pdb else '.'
            intermediates_dir = os.path.join(intermediates_dir, "intermediate_structures")
            os.makedirs(intermediates_dir, exist_ok=True)
        
        # Store individual subunits for merging
        subunit_universes = []
        
        for idx in range(n_subunits):
            # Create a copy of the monomer
            monomer_copy = self.monomer_universe.copy()
            protein = monomer_copy.select_atoms('protein')
            
            # Apply the optimized rotations to the monomer
            if z_rotation != 0.0:
                protein = rotate.rotateby(
                    angle=z_rotation,
                    direction=[0, 0, 1],
                    ag=protein
                )(protein)
            
            if x_rotation != 0.0:
                protein = rotate.rotateby(
                    angle=x_rotation,
                    direction=[1, 0, 0], 
                    ag=protein
                )(protein)
            
            if y_rotation != 0.0:
                protein = rotate.rotateby(
                    angle=y_rotation,
                    direction=[0, 1, 0],
                    ag=protein
                )(protein)
            
            # Calculate position in ring
            angle = 2 * np.pi * idx / n_subunits
            
            # Center the monomer at origin first
            center_of_mass = protein.center_of_mass()
            protein.translate(-center_of_mass)
            
            # Position in ring
            x_pos = radius * np.cos(angle)
            y_pos = radius * np.sin(angle)
            protein.translate([x_pos, y_pos, 0])
            
            # Rotate the subunit to face inward (optional - maintains original orientation)
            # This rotation aligns each subunit with its radial position
            subunit_rotation_angle = np.degrees(angle)
            protein = rotate.rotateby(
                angle=subunit_rotation_angle,
                direction=[0, 0, 1],
                ag=protein
            )(protein)
            
            # Set segment ID
            segid = self.segid_list[idx]
            protein.segments.segids = segid
            
            # Save intermediate file if requested
            if create_intermediates:
                intermediate_file = os.path.join(intermediates_dir, f"subunit_{idx:02d}_{segid}.pdb")
                protein.write(intermediate_file)
            
            # Store for final assembly
            subunit_universes.append(monomer_copy)
        
        # Merge all subunits into final ring
        print(f"Assembling {n_subunits} subunits into ring...")
        
        # Collect all protein AtomGroups
        all_subunit_atoms = []
        for universe in subunit_universes:
            all_subunit_atoms.append(universe.select_atoms('protein'))
        
        # Merge all at once
        ring_system = mda.Merge(*all_subunit_atoms)
        
        # Generate output filename if not provided
        if output_pdb is None:
            output_pdb = f"{n_subunits}mer_r{radius:.1f}A_z{z_rotation:.0f}_x{x_rotation:.0f}_y{y_rotation:.0f}.pdb"
        
        # Write final ring structure
        ring_system.select_atoms('all').write(output_pdb)
        
        print(f"Ring assembly complete: {output_pdb}")
        
        # Print summary
        final_universe = mda.Universe(output_pdb)
        total_atoms = len(final_universe.atoms)
        total_residues = len(final_universe.residues)
        
        print(f"Final structure stats:")
        print(f"  Total atoms: {total_atoms}")
        print(f"  Total residues: {total_residues}")
        print(f"  Segments: {n_subunits} ({', '.join(self.segid_list[:n_subunits])})")
        
        return output_pdb
    
    def build_multiple_rings(self,
                           optimization_results: Dict,
                           subunit_counts: list,
                           output_dir: str = ".",
                           create_intermediates: bool = False) -> Dict[int, str]:
        """
        Build multiple rings with different subunit counts using optimized parameters.
        
        Parameters:
            optimization_results (dict): Results from DimerOptimizer.optimize()
            subunit_counts (list): List of subunit numbers to build
            output_dir (str): Directory for output files
            create_intermediates (bool): Whether to save intermediate files
            
        Returns:
            Dict[int, str]: Mapping of subunit count to output PDB path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        best_params = optimization_results['best_parameters']
        built_rings = {}
        
        print(f"Building multiple rings with optimized parameters:")
        print(f"  Separation distance: {best_params['separation_distance']:.2f} Å")
        print(f"  Rotations: Z={best_params['z_rotation']:.1f}°, X={best_params['x_rotation']:.1f}°, Y={best_params['y_rotation']:.1f}°")
        print(f"  Output directory: {output_dir}")
        
        for n_subunits in subunit_counts:
            print(f"\n--- Building {n_subunits}-mer ---")
            
            # Calculate radius for this ring size
            radius = self.calculate_ring_radius(best_params['separation_distance'], n_subunits)
            
            # Generate output filename
            output_pdb = os.path.join(
                output_dir, 
                f"{n_subunits}mer_optimized_r{radius:.1f}A.pdb"
            )
            
            try:
                final_pdb = self.build_ring(
                    n_subunits=n_subunits,
                    separation_distance=best_params['separation_distance'],
                    z_rotation=best_params['z_rotation'],
                    x_rotation=best_params['x_rotation'],
                    y_rotation=best_params['y_rotation'],
                    output_pdb=output_pdb,
                    create_intermediates=create_intermediates
                )
                built_rings[n_subunits] = final_pdb
                
            except Exception as e:
                print(f"Error building {n_subunits}-mer: {e}")
                
        print(f"\n--- Summary ---")
        print(f"Successfully built {len(built_rings)} rings:")
        for n_subunits, pdb_path in built_rings.items():
            print(f"  {n_subunits}-mer: {pdb_path}")
            
        return built_rings


def main():
    """Main function for command-line usage."""
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Build circular assemblies using optimized parameters')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--optimization_results', help='CSV file with optimization results (fine_search_results.csv)')
    parser.add_argument('--n_subunits', type=int, help='Number of subunits for single ring')
    parser.add_argument('--subunit_range', type=int, nargs=2, help='Range of subunit numbers (min max)')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    parser.add_argument('--create_intermediates', action='store_true', help='Save intermediate subunit files')
    
    # Manual parameter specification (alternative to optimization file)
    parser.add_argument('--separation_distance', type=float, help='Manual: separation distance')
    parser.add_argument('--z_rotation', type=float, default=0, help='Manual: Z rotation (degrees)')
    parser.add_argument('--x_rotation', type=float, default=0, help='Manual: X rotation (degrees)')
    parser.add_argument('--y_rotation', type=float, default=0, help='Manual: Y rotation (degrees)')
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = OptimizedRingBuilder(args.monomer)
    
    # Load optimization results or use manual parameters
    if args.optimization_results:
        print(f"Loading optimization results from {args.optimization_results}")
        results_df = pd.read_csv(args.optimization_results)
        best_params = results_df.iloc[0].to_dict()  # Best result (lowest score)
        
        optimization_results = {'best_parameters': best_params}
        
        if args.n_subunits:
            # Build single ring
            output_pdb = os.path.join(args.output_dir, f"{args.n_subunits}mer_optimized.pdb")
            builder.build_ring_from_optimization(
                optimization_results, args.n_subunits, output_pdb, args.create_intermediates
            )
        elif args.subunit_range:
            # Build multiple rings
            subunit_counts = list(range(args.subunit_range[0], args.subunit_range[1] + 1))
            builder.build_multiple_rings(
                optimization_results, subunit_counts, args.output_dir, args.create_intermediates
            )
        else:
            print("Error: Specify either --n_subunits or --subunit_range")
            
    elif all([args.separation_distance is not None, args.n_subunits]):
        # Manual parameters
        print("Using manual parameters")
        output_pdb = os.path.join(args.output_dir, f"{args.n_subunits}mer_manual.pdb")
        builder.build_ring(
            n_subunits=args.n_subunits,
            separation_distance=args.separation_distance,
            z_rotation=args.z_rotation,
            x_rotation=args.x_rotation,
            y_rotation=args.y_rotation,
            output_pdb=output_pdb,
            create_intermediates=args.create_intermediates
        )
    else:
        print("Error: Provide either --optimization_results or manual parameters (--separation_distance)")
        print("Use --help for usage information")


if __name__ == "__main__":
    main()