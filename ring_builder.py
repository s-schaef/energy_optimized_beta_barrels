#!/usr/bin/env python3

import os
import numpy as np
import MDAnalysis as mda
import pyrosetta
from pyrosetta import rosetta
from MDAnalysis.transformations import rotate
from string import ascii_uppercase as auc
from string import ascii_lowercase as alc
from typing import Dict, Optional
import tempfile


class RingBuilder:
    """
    Build circular protein assemblies from aligned monomer structures.
    
    The monomer should be pre-aligned so that the desired interface faces
    outward when positioned in the ring.
    """
    
    def __init__(self, monomer_pdb: str):
        """
        Initialize ring builder with aligned monomer.
        
        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
            
        Raises:
            FileNotFoundError: If monomer PDB file doesn't exist
            ValueError: If PDB file contains no protein atoms
        """
        if not os.path.exists(monomer_pdb):
            raise FileNotFoundError(f"Monomer PDB file not found: {monomer_pdb}")
            
        self.monomer_pdb = monomer_pdb
        self.monomer_universe = mda.Universe(monomer_pdb)
        self.monomer_atoms = self.monomer_universe.select_atoms('protein')
        
        if len(self.monomer_atoms) == 0:
            raise ValueError(f"No protein atoms found in {monomer_pdb}")
        
        self.ring = None
        self.scorefxn = None
        self.output_pdb = None
        
        print(f"Initialized RingBuilder with monomer containing {len(self.monomer_atoms)} protein atoms")

    def _initialize_pyrosetta(self):
        """Initialize PyRosetta with appropriate scoring function."""
        if self.scorefxn is not None:
            return  # Already initialized
            
        pyrosetta.init('-mute all')
        
        self.scorefxn = pyrosetta.create_score_function('empty')
        self.scorefxn.set_weight(rosetta.core.scoring.fa_atr, 1) # downscaled to give more importance to hbonds
        self.scorefxn.set_weight(rosetta.core.scoring.fa_rep, 0.01) # downscaled to give more importance to hbonds
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_sr_bb, 1) # short-range backbone hydrogen bonds
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_lr_bb, 10.0) # long-range backbone hydrogen bonds
        
        print("PyRosetta initialized with scoring terms: fa_atr, fa_rep, hbond_sr_bb, hbond_lr_bb")
    
    def build_ring(self, n_subunits: int = 30, radius: float = 50.0, tilt_angle: float = 0.0):
        """
        Build a circular assembly of protein subunits.
        
        Parameters:
            n_subunits (int): Number of subunits in the ring (default: 30)
            radius (float): Radius of the circular assembly in Angstroms (default: 50.0)
            tilt_angle (float): Tilt angle of the subunit in degrees (default: 0.0)
            
        Returns:
            MDAnalysis.Universe: The assembled ring structure
            
        Raises:
            ValueError: If n_subunits < 2 or radius <= 0
        """
        if n_subunits < 2:
            raise ValueError("n_subunits must be at least 2")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if n_subunits > 52:  # len(auc + alc)
            raise ValueError("n_subunits cannot exceed 52 (limited by available segment IDs)")
            
        tmp_universes = []
        segid_list = (auc + alc)  # Create a list of segment IDs (A-Z, a-z)

        for idx in range(n_subunits):
            # Create a copy of the monomer
            subunit = self.monomer_universe.copy()
            protein = subunit.select_atoms('protein')
            
            # Apply tilt angle around x-axis (for beta-sheet orientation)
            if tilt_angle != 0.0:
                protein.rotateby(tilt_angle, axis=[1, 0, 0])

            # Rotate subunit to face the center when it is later positioned in the ring
            angle = 360 * idx / n_subunits
            protein.rotateby(angle, axis=[0, 0, 1])


            # Calculate position in ring
            x_pos = radius * np.cos(np.radians(angle))
            y_pos = radius * np.sin(np.radians(angle))
            
            # Move subunit to its position in the ring
            protein.translate([x_pos, y_pos, 0])
            
            # Rotate subunit to face the center

            # Assign unique segment ID
            protein.segments.segids = segid_list[idx]
            tmp_universes.append(protein)

        # Combine all subunits into a single universe
        self.ring = mda.Merge(*[u.atoms for u in tmp_universes])
        
        print(f"Built ring with {n_subunits} subunits, radius {radius:.1f} Å, tilt angle {tilt_angle:.1f}°")
        return self.ring
    
    def center_ring(self):
        """
        Center the ring assembly at the origin.
        
        Raises:
            RuntimeError: If ring hasn't been built yet
        """
        if self.ring is None:
            raise RuntimeError("Ring must be built before centering. Call build_ring() first.")
            
        center_of_mass = self.ring.atoms.center_of_mass()
        self.ring.atoms.translate(-center_of_mass)
        print(f"Ring centered at origin (was at {center_of_mass})")
    
    def write_ring_pdb(self, output_pdb: str, centered: bool = True):
        """
        Write the assembled ring to a PDB file.
        
        Parameters:
            output_pdb (str): Path for output PDB file
            centered (bool): Whether to center the ring at origin (default: True)
            
        Raises:
            RuntimeError: If ring hasn't been built yet
        """
        if self.ring is None:
            raise RuntimeError("Ring must be built before writing. Call build_ring() first.")

        if centered:
            self.center_ring()

        # # Ensure output directory exists (only if there's a directory component) #TODO: was this needed?
        # output_dir = os.path.dirname(output_pdb)
        # if output_dir:  # Only create directory if dirname returns something
        #     os.makedirs(output_dir, exist_ok=True)
        
        self.ring.atoms.write(output_pdb)
        print(f"Ring assembly written to {output_pdb}")

    # def evaluate_geometry(self, n_subunits: int, radius: float, tilt_angle: float) -> Dict[str, float]:
    #     """
    #     Build and score a ring with given parameters.
        
    #     Parameters:
    #         n_subunits (int): Number of subunits
    #         radius (float): Ring radius
    #         tilt_angle (float): Tilt angle
            
    #     Returns:
    #         Dict[str, float]: Score dictionary with geometry parameters
    #     """
    #     # Build the ring
    #     self.build_ring(n_subunits=n_subunits, radius=radius, tilt_angle=tilt_angle)
        
    #     # Score it
    #     scores = self.score_ring()
        
    #     # Add geometry parameters
    #     scores.update({
    #         'n_subunits': n_subunits,
    #         'radius': radius,
    #         'tilt_angle': tilt_angle
    #     })
        
    #     return scores

    # def score_ring(self) -> Dict[str, float]:
    #     """
    #     Score the assembled ring using PyRosetta.
        
    #     Returns:
    #         Dict[str, float]: Dictionary containing individual score components
            
    #     Raises:
    #         RuntimeError: If ring hasn't been built yet
    #     """
    #     if self.ring is None:
    #         raise RuntimeError("Ring must be built before scoring. Call build_ring() first.")
            
    #     if not hasattr(self, 'scorefxn') or self.scorefxn is None:
    #         self._initialize_pyrosetta()
        
    #     # Create temporary file for scoring
    #     with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
    #         temp_pdb = tmp.name
        
    #     try:
    #         self.write_ring_pdb(temp_pdb, centered=True)
            
    #         # Create a PyRosetta pose from the ring
    #         pose = pyrosetta.pose_from_pdb(temp_pdb)
            
    #         # Get total score
    #         total_score = self.scorefxn(pose)
            
    #         # Get individual score components
    #         score_dict = {
    #             'total_score': total_score,
    #             'fa_atr': self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.fa_atr),
    #             'fa_rep': self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.fa_rep),
    #             'hbond_sr_bb': self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.hbond_sr_bb),
    #             'hbond_lr_bb': self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.hbond_lr_bb),
    #             'n_subunits': len(self.ring.segments)  # Add this for tracking
    #         }
            
    #         return score_dict
            
    #     finally:
    #         # Clean up temporary file
    #         if os.path.exists(temp_pdb):
    #             os.remove(temp_pdb)

    def score_ring(self) -> Dict[str, float]:
        """
        Score the assembled ring using PyRosetta.
        
        Returns:
            Dict[str, float]: Dictionary containing individual score components
            
        Raises:
            RuntimeError: If ring hasn't been built yet
        """
        if self.ring is None:
            raise RuntimeError("Ring must be built before scoring. Call build_ring() first.")
            
        if not hasattr(self, 'scorefxn') or self.scorefxn is None:
            self._initialize_pyrosetta()
        
        # Create a unique temporary file for this specific scoring operation
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
            temp_pdb_path = tmp.name
        
        try:
            # Write ring to the temporary file
            self.write_ring_pdb(temp_pdb_path, centered=True)
            
            # Create a PyRosetta pose from the ring
            pose = pyrosetta.pose_from_pdb(temp_pdb_path)
            
            # Get total score
            total_score = self.scorefxn(pose)
            
            # Get individual score components
            score_dict = {
                'total_score': total_score,
                'fa_atr': self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.fa_atr),
                'fa_rep': self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.fa_rep),
                'hbond_sr_bb': self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.hbond_sr_bb),
                'hbond_lr_bb': self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.hbond_lr_bb),
                'n_subunits': len(self.ring.segments)
            }
            
            return score_dict
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_pdb_path):
                os.remove(temp_pdb_path)

    def get_ring_info(self):
        """
        Get information about the current ring assembly.
        
        Returns:
            dict: Information about the ring (or None if not built)
        """
        if self.ring is None:
            return None
            
        return {
            'n_atoms': len(self.ring.atoms),
            'n_residues': len(self.ring.residues),
            'n_segments': len(self.ring.segments),
            'center_of_mass': self.ring.atoms.center_of_mass(),
            'dimensions': self.ring.dimensions
        }


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build a circular protein assembly from an aligned monomer PDB file.')
    parser.add_argument('--input', required=True, help='Monomeric input PDB file')
    parser.add_argument('--output', required=True, help='Circular output PDB file')
    parser.add_argument('--n_subunits', type=int, default=30, help='Number of subunits in the ring (default: 30)')
    parser.add_argument('--score', action='store_true', help='Score the assembly with PyRosetta')
    
    args = parser.parse_args()
    
    try:
        # Initialize the ring builder
        ring_builder = RingBuilder(args.input)
        
        # Build the ring
        ring_builder.build_ring(args.n_subunits, args.radius, args.tilt_angle)
        
        # Write the result
        if not args.output == None:
            ring_builder.write_ring_pdb(args.output)
            ring_builder.output_pdb = args.output
        
        # Score if requested
        if args.score:
            score = ring_builder.score_ring()
            print(f"Final energy: {score:.2f}")
            
        # Print ring information
        info = ring_builder.get_ring_info()
        if info:
            print(f"Ring info: {info['n_atoms']} atoms, {info['n_residues']} residues, {info['n_segments']} segments")
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)