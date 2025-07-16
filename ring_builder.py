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
        
        print(f"Initialized RingBuilder with monomer containing {len(self.monomer_atoms)} protein atoms")

    def _initialize_pyrosetta(self):
        """Initialize PyRosetta with appropriate scoring function."""
        if self.scorefxn is not None:
            return  # Already initialized
            
        pyrosetta.init('-mute all')
        
        self.scorefxn = pyrosetta.create_score_function('empty')
        self.scorefxn.set_weight(rosetta.core.scoring.fa_atr, 1.0)
        self.scorefxn.set_weight(rosetta.core.scoring.fa_rep, 1.0)
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_sr_bb, 1.0) # short-range backbone hydrogen bonds
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_lr_bb, 1.0) # long-range backbone hydrogen bonds
        
        print("PyRosetta initialized with scoring terms: fa_atr, fa_rep, hbond_bb_sc")

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

        for idx in range(self.n_subunits):
            subunit = self.monomer_universe.copy()
            protein = subunit.select_atoms('protein')

            # Apply tilt angle around y-axis (for beta-sheet orientation)
            if tilt_angle != 0.0:
                protein.rotateby(tilt_angle, axis=[0, 1, 0])

            # Calculate position in ring
            angle = 2 * np.pi * idx / n_subunits
            x_pos = radius * np.cos(angle)
            y_pos = radius * np.sin(angle)

            # Move subunit to its position in the ring
            protein.translate([x_pos, y_pos, 0])
            
            # Rotate subunit to face the center
            protein.rotateby(np.degrees(angle), axis=[0, 0, 1])

            # add subunit to the list of universes
            protein.segments.segids = segid_list[idx]
            tmp_universes.append(protein)



        # Combine all subunits into a single universe
        # Merge all universes into one
        ring = mda.Merge(*[u.atoms for u in tmp_universes])

        return ring
    
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
            
            # Apply tilt angle around y-axis (for beta-sheet orientation)
            if tilt_angle != 0.0:
                protein.rotateby(tilt_angle, axis=[0, 1, 0])

            # Rotate subunit to face the center when it is later positioned in the ring
            angle = 360 * idx / n_subunits
            protein.rotateby(angle-90, axis=[0, 0, 1])


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

        # Ensure output directory exists (only if there's a directory component)
        output_dir = os.path.dirname(output_pdb)
        if output_dir:  # Only create directory if dirname returns something
            os.makedirs(output_dir, exist_ok=True)
        
        self.ring.atoms.write(output_pdb)
        print(f"Ring assembly written to {output_pdb}")

    def score_ring(self, pdb_file: Optional[str] = None):
        """
        Score the assembled ring using PyRosetta.
        
        Parameters:
            pdb_file (str, optional): PDB file to score. If None, creates temporary file.
            
        Returns:
            float: PyRosetta energy score
            
        Raises:
            RuntimeError: If ring hasn't been built yet
        """
        if self.ring is None:
            raise RuntimeError("Ring must be built before scoring. Call build_ring() first.")
            
        if not hasattr(self, 'scorefxn') or self.scorefxn is None:
            self._initialize_pyrosetta()
        
        # Write to temporary file if no file specified
        temp_file = False
        if pdb_file is None:
            pdb_file = os.path.join(tempfile.gettempdir(), 'temp_ring_for_scoring.pdb')
            temp_file = True
        
        self.write_ring_pdb(pdb_file, centered=True)
        
        try:
            # Create a PyRosetta pose from the ring
            pose = pyrosetta.pose_from_pdb(pdb_file)
            
            # Score the pose
            score = self.scorefxn(pose)
            print(f"Ring assembly score: {score:.2f}")
            
            return score
            
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(pdb_file):
                os.remove(pdb_file)

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
    parser.add_argument('--radius', type=float, default=80.0, help='Radius of the circular assembly (default: 80.0)')
    parser.add_argument('--tilt_angle', type=float, default=0.0, help='Tilt angle of the beta-sheet in degrees (default: 0.0)')
    parser.add_argument('--score', action='store_true', help='Score the assembly with PyRosetta')
    
    args = parser.parse_args()
    
    try:
        # Initialize the ring builder
        ring_builder = RingBuilder(args.input)
        
        # Build the ring
        ring_builder.build_ring(args.n_subunits, args.radius, args.tilt_angle)
        
        # Write the result
        ring_builder.write_ring_pdb(args.output)
        
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