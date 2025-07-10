#!/usr/bin/env python3

import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import rotate
import pyrosetta
from pyrosetta import rosetta
import os
import tempfile
from typing import Tuple, Dict, Optional

class DimerBuilder:
    """
    Build and score protein dimers to find optimal geometry parameters
    for circular assembly construction.
    """
    
    def __init__(self, monomer_pdb: str, initialize_pyrosetta: bool = True):
        """
        Initialize dimer builder with aligned monomer.
        
        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
            initialize_pyrosetta (bool): Whether to initialize PyRosetta
        """
        self.monomer_pdb = monomer_pdb
        self.monomer_universe = mda.Universe(monomer_pdb)
        self.monomer_atoms = self.monomer_universe.select_atoms('protein')
        
        if initialize_pyrosetta:
            self._initialize_pyrosetta()
    
    def _initialize_pyrosetta(self):
        """Initialize PyRosetta with appropriate scoring function."""
        # Initialize PyRosetta (suppress output)
        pyrosetta.init('-mute all')
        
        # Create scoring function with only the terms we want
        self.scorefxn = pyrosetta.create_score_function('empty')
        
        # Add the specific terms we discussed
        self.scorefxn.set_weight(rosetta.core.scoring.fa_atr, 1.0)      # attractive VdW
        self.scorefxn.set_weight(rosetta.core.scoring.fa_rep, 1.0)      # repulsive VdW  
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_bb_sc, 1.0)  # backbone-sidechain H-bonds
        
        print("PyRosetta initialized with scoring terms:")
        print("  fa_atr: attractive van der Waals")
        print("  fa_rep: repulsive van der Waals") 
        print("  hbond_bb_sc: backbone-sidechain hydrogen bonds")
    
    def build_dimer(self, 
                   separation_distance: float,
                   z_rotation: float = 0.0,
                   x_rotation: float = 0.0, 
                   y_rotation: float = 0.0,
                   output_pdb: Optional[str] = None) -> str:
        """
        Build a dimer with specified geometry parameters.
        
        This tests different monomer orientations by applying the same rotations
        to both monomers, exploring whether this orientation is optimal for 
        circular assembly.
        
        Parameters:
            separation_distance (float): Distance between monomer centers (Angstroms)
            z_rotation (float): Rotation around z-axis applied to both monomers (degrees)
            x_rotation (float): Rotation around x-axis applied to both monomers (degrees)  
            y_rotation (float): Rotation around y-axis applied to both monomers (degrees)
            output_pdb (str, optional): Output PDB filename
            
        Returns:
            str: Path to dimer PDB file
        """
        # Create two copies of the monomer
        monomer1 = self.monomer_universe.copy()
        monomer2 = self.monomer_universe.copy()
        
        # Get protein atoms
        protein1 = monomer1.select_atoms('protein')
        protein2 = monomer2.select_atoms('protein')
        
        # Apply the same rotations to both monomers first
        if z_rotation != 0.0:
            protein1 = rotate.rotateby(
                angle=z_rotation, 
                direction=[0, 0, 1], 
                ag=protein1
            )(protein1)
            protein2 = rotate.rotateby(
                angle=z_rotation, 
                direction=[0, 0, 1], 
                ag=protein2
            )(protein2)
        
        if x_rotation != 0.0:
            # Rotate around x-axis
            protein1 = rotate.rotateby(
                angle=x_rotation,
                direction=[1, 0, 0],
                ag=protein1
            )(protein1)
            protein2 = rotate.rotateby(
                angle=x_rotation,
                direction=[1, 0, 0],
                ag=protein2
            )(protein2)
        
        if y_rotation != 0.0:
            # Rotate around y-axis
            protein1 = rotate.rotateby(
                angle=y_rotation,
                direction=[0, 1, 0],
                ag=protein1
            )(protein1)
            protein2 = rotate.rotateby(
                angle=y_rotation,
                direction=[0, 1, 0],
                ag=protein2
            )(protein2)
        
        # Now position the rotated monomers
        # Position monomer1 at -separation_distance/2 on x-axis
        protein1.translate([-separation_distance/2, 0, 0])
        
        # Position monomer2 at +separation_distance/2 on x-axis  
        protein2.translate([separation_distance/2, 0, 0])
        
        # Merge the two monomers
        dimer = mda.Merge(protein1, protein2)
        
        # Update segment IDs to distinguish monomers
        dimer.segments[0].segid = 'A'
        dimer.segments[1].segid = 'B'
        
        # Write dimer to file
        if output_pdb is None:
            # Create temporary file
            temp_fd, output_pdb = tempfile.mkstemp(suffix='.pdb', prefix='dimer_')
            os.close(temp_fd)
        
        dimer.select_atoms('all').write(output_pdb)
        
        return output_pdb
    
    def score_dimer(self, dimer_pdb: str) -> Dict[str, float]:
        """
        Score a dimer using PyRosetta.
        
        Parameters:
            dimer_pdb (str): Path to dimer PDB file
            
        Returns:
            Dict[str, float]: Dictionary with score components and total score
        """
        # Load structure into PyRosetta
        pose = pyrosetta.pose_from_pdb(dimer_pdb)
        
        # Calculate total score
        total_score = self.scorefxn(pose)
        
        # Get individual score components
        scores = {
            'total_score': total_score,
            'fa_atr': pose.energies().total_energies()[rosetta.core.scoring.fa_atr],
            'fa_rep': pose.energies().total_energies()[rosetta.core.scoring.fa_rep],
            'hbond_bb_sc': pose.energies().total_energies()[rosetta.core.scoring.hbond_bb_sc]
        }
        
        return scores
    
    def evaluate_geometry(self, 
                         separation_distance: float,
                         z_rotation: float = 0.0,
                         x_rotation: float = 0.0,
                         y_rotation: float = 0.0,
                         cleanup: bool = True) -> Dict[str, float]:
        """
        Build and score a dimer with given geometry parameters.
        
        Parameters:
            separation_distance (float): Distance between monomer centers
            z_rotation (float): Rotation around z-axis (degrees)
            x_rotation (float): Rotation around x-axis (degrees)
            y_rotation (float): Rotation around y-axis (degrees)
            cleanup (bool): Whether to delete temporary PDB file
            
        Returns:
            Dict[str, float]: Score dictionary with geometry parameters added
        """
        # Build dimer
        dimer_pdb = self.build_dimer(
            separation_distance=separation_distance,
            z_rotation=z_rotation,
            x_rotation=x_rotation,
            y_rotation=y_rotation
        )
        
        # Score dimer
        scores = self.score_dimer(dimer_pdb)
        
        # Add geometry parameters to results
        scores.update({
            'separation_distance': separation_distance,
            'z_rotation': z_rotation,
            'x_rotation': x_rotation,
            'y_rotation': y_rotation
        })
        
        # Cleanup temporary file
        if cleanup and os.path.exists(dimer_pdb):
            os.remove(dimer_pdb)
        
        return scores
    
    def calculate_ring_radius(self, separation_distance: float, n_subunits: int) -> float:
        """
        Calculate the radius of a circular ring given the dimer separation distance.
        
        For a regular polygon, the radius is related to the side length by:
        R = (side_length / 2) / sin(π/n)
        
        Parameters:
            separation_distance (float): Optimal separation distance from dimer
            n_subunits (int): Number of subunits in the final ring
            
        Returns:
            float: Radius of the circular ring
        """
        angle_between_subunits = 2 * np.pi / n_subunits
        radius = separation_distance / (2 * np.sin(angle_between_subunits / 2))
        return radius
    
    def get_ring_geometry(self, 
                         separation_distance: float, 
                         n_subunits: int,
                         z_rotation: float = 0.0,
                         x_rotation: float = 0.0,
                         y_rotation: float = 0.0) -> Dict[str, float]:
        """
        Convert dimer geometry to ring building parameters.
        
        Parameters:
            separation_distance (float): Optimal separation from dimer optimization
            n_subunits (int): Number of subunits in target ring
            z_rotation (float): Optimal z-rotation from dimer optimization
            x_rotation (float): Optimal x-rotation from dimer optimization
            y_rotation (float): Optimal y-rotation from dimer optimization
            
        Returns:
            Dict[str, float]: Ring building parameters
        """
        radius = self.calculate_ring_radius(separation_distance, n_subunits)
        
        return {
            'radius': radius,
            'z_rotation': z_rotation,
            'x_rotation': x_rotation,
            'y_rotation': y_rotation,
            'separation_distance': separation_distance,
            'n_subunits': n_subunits
        }


def test_dimer_builder(monomer_pdb: str, output_dir: str = '.'):
    """
    Test function to build and score a few dimers with different parameters.
    
    Parameters:
        monomer_pdb (str): Path to aligned monomer PDB
        output_dir (str): Directory for output files
    """
    print(f"Testing dimer builder with monomer: {monomer_pdb}")
    
    # Initialize builder
    builder = DimerBuilder(monomer_pdb)
    
    # Test a few different geometries
    test_cases = [
        {'separation_distance': 10.0, 'z_rotation': 0.0},
        {'separation_distance': 12.0, 'z_rotation': 0.0},
        {'separation_distance': 15.0, 'z_rotation': 0.0},
        {'separation_distance': 12.0, 'z_rotation': 180.0},
        {'separation_distance': 12.0, 'z_rotation': 0.0, 'x_rotation': 10.0},
    ]
    
    print("\nTesting different dimer geometries:")
    print("=" * 60)
    
    for i, params in enumerate(test_cases):
        print(f"\nTest {i+1}: {params}")
        
        # Build and score dimer
        scores = builder.evaluate_geometry(**params, cleanup=False)
        
        # Print results
        print(f"  Total score: {scores['total_score']:.2f}")
        print(f"  fa_atr: {scores['fa_atr']:.2f}")
        print(f"  fa_rep: {scores['fa_rep']:.2f}")
        print(f"  hbond_bb_sc: {scores['hbond_bb_sc']:.2f}")
        
        # Calculate what this would mean for different ring sizes
        for n_subunits in [50, 51, 52]:
            ring_params = builder.get_ring_geometry(
                scores['separation_distance'], 
                n_subunits,
                scores['z_rotation'],
                scores['x_rotation'],
                scores['y_rotation']
            )
            print(f"  → {n_subunits}-mer ring radius: {ring_params['radius']:.1f} Å")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dimer builder')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    
    args = parser.parse_args()
    
    test_dimer_builder(args.monomer, args.output_dir)