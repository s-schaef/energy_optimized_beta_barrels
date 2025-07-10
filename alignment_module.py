#!/usr/bin/env python3

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.transformations import rotate
from sklearn.decomposition import PCA
import os

class MonomerAligner:
    """
    Class to standardize monomer orientation for consistent dimer optimization.
    Aligns protein's principal axis with z-axis and centers at origin.
    """
    
    def __init__(self, pdb_file):
        """
        Initialize with a PDB file containing a single monomer.
        
        Parameters:
            pdb_file (str): Path to PDB file with monomer structure
        """
        self.universe = mda.Universe(pdb_file)
        self.protein = self.universe.select_atoms('protein')
        
    def align_to_standard_orientation(self, output_file=None):
        """
        Align monomer to standard orientation:
        1. Center at origin (0,0,0)
        2. Align principal axis with z-axis
        
        Parameters:
            output_file (str, optional): If provided, write aligned structure to this file
            
        Returns:
            MDAnalysis.AtomGroup: Aligned protein atoms
        """
        # Step 1: Center the protein at origin
        center_of_mass = self.protein.center_of_mass()
        self.protein.translate(-center_of_mass)
        
        # Step 2: Calculate principal axes using PCA on CA atoms
        ca_atoms = self.protein.select_atoms('name CA')
        if len(ca_atoms) == 0:
            raise ValueError("No CA atoms found in protein structure")
            
        ca_positions = ca_atoms.positions
        
        # Perform PCA to find principal axes
        pca = PCA(n_components=3)
        pca.fit(ca_positions)
        
        # Get the principal axes (eigenvectors)
        principal_axes = pca.components_
        
        # The first principal axis should be aligned with z-axis
        primary_axis = principal_axes[0]
        
        # Calculate rotation to align primary axis with z-axis
        z_axis = np.array([0, 0, 1])
        
        # Calculate rotation axis (cross product)
        rotation_axis = np.cross(primary_axis, z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # If axes are already aligned (or anti-aligned), no rotation needed
        if rotation_axis_norm < 1e-6:
            # Check if anti-aligned (dot product < 0)
            if np.dot(primary_axis, z_axis) < 0:
                # Rotate 180 degrees around x-axis
                rotation_axis = np.array([1, 0, 0])
                rotation_angle = 180.0
            else:
                rotation_angle = 0.0
        else:
            # Normalize rotation axis
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # Calculate rotation angle
            cos_angle = np.dot(primary_axis, z_axis)
            rotation_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            rotation_angle = np.degrees(rotation_angle)
        
        # Apply rotation if needed
        if rotation_angle > 1e-6:
            self.protein = rotate.rotateby(
                angle=rotation_angle, 
                direction=rotation_axis, 
                ag=self.protein
            )(self.protein)
        
        # Step 3: Optional secondary alignment to standardize orientation around z-axis
        # This ensures consistent orientation for dimer building
        self._standardize_z_orientation()
        
        # Write output if requested
        if output_file:
            self.protein.write(output_file)
            print(f"Aligned monomer written to {output_file}")
            
        return self.protein
    
    def _standardize_z_orientation(self):
        """
        Standardize the orientation around the z-axis by aligning the second
        principal axis with the x-axis (in the xy-plane).
        """
        # Get CA atoms for PCA
        ca_atoms = self.protein.select_atoms('name CA')
        ca_positions = ca_atoms.positions
        
        # Perform PCA again after z-alignment
        pca = PCA(n_components=3)
        pca.fit(ca_positions)
        
        # Get the second principal axis (should be in xy-plane now)
        second_axis = pca.components_[1]
        
        # Project onto xy-plane and normalize
        second_axis_xy = second_axis[:2]
        second_axis_xy = second_axis_xy / np.linalg.norm(second_axis_xy)
        
        # Calculate angle to align with x-axis
        x_axis = np.array([1, 0])
        cos_angle = np.dot(second_axis_xy, x_axis)
        sin_angle = np.cross(second_axis_xy, x_axis)
        
        rotation_angle = np.arctan2(sin_angle, cos_angle)
        rotation_angle = np.degrees(rotation_angle)
        
        # Rotate around z-axis
        if abs(rotation_angle) > 1e-6:
            self.protein = rotate.rotateby(
                angle=rotation_angle,
                direction=[0, 0, 1],
                ag=self.protein
            )(self.protein)
    
    def get_aligned_monomer(self):
        """
        Get the aligned monomer atom group.
        
        Returns:
            MDAnalysis.AtomGroup: Aligned protein atoms
        """
        return self.protein
    
    def get_dimensions(self):
        """
        Get the dimensions of the aligned monomer.
        
        Returns:
            dict: Dictionary with min/max coordinates and center of mass
        """
        positions = self.protein.positions
        return {
            'x_min': positions[:, 0].min(),
            'x_max': positions[:, 0].max(),
            'y_min': positions[:, 1].min(),
            'y_max': positions[:, 1].max(),
            'z_min': positions[:, 2].min(),
            'z_max': positions[:, 2].max(),
            'center_of_mass': self.protein.center_of_mass()
        }


def align_monomer_from_file(input_pdb, output_pdb=None):
    """
    Convenience function to align a monomer from a PDB file.
    
    Parameters:
        input_pdb (str): Path to input PDB file
        output_pdb (str, optional): Path to output aligned PDB file
        
    Returns:
        MonomerAligner: Aligner object with aligned monomer
    """
    aligner = MonomerAligner(input_pdb)
    aligner.align_to_standard_orientation(output_pdb)
    
    # Print alignment summary
    dims = aligner.get_dimensions()
    print(f"Monomer alignment complete:")
    print(f"  Center of mass: {dims['center_of_mass']}")
    print(f"  X range: {dims['x_min']:.2f} to {dims['x_max']:.2f}")
    print(f"  Y range: {dims['y_min']:.2f} to {dims['y_max']:.2f}")
    print(f"  Z range: {dims['z_min']:.2f} to {dims['z_max']:.2f}")
    
    return aligner


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Align monomer to standard orientation\
                                     (1st principal axis with z-axis,\
                                      2nd principal axis with x-axis,\
                                      centered at origin)')
    parser.add_argument('--input', required=True, help='Input PDB file')
    parser.add_argument('--output', help='Output PDB file (optional)')
    
    args = parser.parse_args()
    
    # Align the monomer
    aligner = align_monomer_from_file(args.input, args.output)