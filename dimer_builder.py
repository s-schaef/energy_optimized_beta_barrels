#!/usr/bin/env python3

import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import rotate
import pyrosetta
from pyrosetta import rosetta
import os
import tempfile
from typing import Tuple, Dict, Optional
from sklearn.decomposition import PCA
import itertools

class DimerBuilder:
    """
    Build and score protein dimers to find optimal geometry parameters
    for circular assembly construction using beta-sheet edge-to-edge positioning.
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
        
        # Detect beta-sheet region and edge direction
        self.beta_sheet_info = self._detect_beta_sheet_region()
        print(f"Detected beta-sheet: {len(self.beta_sheet_info['residues'])} residues")
        print(f"Beta-sheet direction: [{self.beta_sheet_info['direction'][0]:.3f}, {self.beta_sheet_info['direction'][1]:.3f}]")
    
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
    
    def _detect_beta_sheet_region(self) -> Dict:
        """
        Detect the main beta-sheet region as the largest rectangular area of 'E' residues.
        
        Returns:
            Dict: Information about the beta-sheet including residues and edge direction
        """
        try:
            # Use PyRosetta to assign secondary structure
            pose = pyrosetta.pose_from_pdb(self.monomer_pdb)
            
            # Get secondary structure using DSSP
            from pyrosetta.rosetta.core.scoring.dssp import Dssp
            dssp = Dssp(pose)
            dssp.insert_ss_into_pose(pose)
            
            # Get all residues with their positions and secondary structure
            residue_info = []
            for i in range(1, pose.total_residue() + 1):  # Rosetta uses 1-based indexing
                residue = pose.residue(i)
                ss = pose.secstruct(i)
                ca_atom = residue.atom("CA")
                residue_info.append({
                    'residue_num': i,
                    'ss': ss,
                    'pos': np.array([ca_atom.xyz().x, ca_atom.xyz().y, ca_atom.xyz().z])
                })
            
            # Find beta-strand residues
            beta_residues = [r for r in residue_info if r['ss'] == 'E']
            
            if len(beta_residues) < 3:
                print("Warning: Less than 3 beta-strand residues found. Using fallback method.")
                return self._fallback_beta_sheet_detection()
            
            # Find the largest rectangular region of beta-residues
            beta_sheet_region = self._find_largest_beta_rectangle(beta_residues)
            
            return beta_sheet_region
            
        except Exception as e:
            print(f"Error in secondary structure detection: {e}")
            print("Using fallback beta-sheet detection method.")
            return self._fallback_beta_sheet_detection()
    
    def _find_largest_beta_rectangle(self, beta_residues: list) -> Dict:
        """
        Find the largest rectangular region of beta-strand residues.
        
        Parameters:
            beta_residues (list): List of beta-strand residue information
            
        Returns:
            Dict: Beta-sheet region information
        """
        if len(beta_residues) < 3:
            return self._fallback_beta_sheet_detection()
        
        # Get CA positions of beta residues (only xy coordinates)
        beta_positions = np.array([r['pos'][:2] for r in beta_residues])
        
        # Use PCA to find the main directions of the beta-sheet
        pca = PCA(n_components=2)
        pca.fit(beta_positions)
        
        # Transform positions to PCA coordinate system
        transformed_positions = pca.transform(beta_positions)
        
        # Find the largest rectangular region in PCA space
        # This represents the main beta-sheet area
        x_min, x_max = transformed_positions[:, 0].min(), transformed_positions[:, 0].max()
        y_min, y_max = transformed_positions[:, 1].min(), transformed_positions[:, 1].max()
        
        # Define the rectangle boundaries (use 80% of the range to avoid outliers)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        x_bounds = [x_center - 0.4 * x_range, x_center + 0.4 * x_range]
        y_bounds = [y_center - 0.4 * y_range, y_center + 0.4 * y_range]
        
        # Find residues within this rectangle
        sheet_residues = []
        for i, (x, y) in enumerate(transformed_positions):
            if x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]:
                sheet_residues.append(beta_residues[i])
        
        if len(sheet_residues) < 3:
            sheet_residues = beta_residues  # Use all beta residues if rectangle is too small
        
        # The sheet direction is the first principal component (longest axis)
        sheet_direction_3d = pca.components_[0]
        
        # Project to xy-plane
        sheet_direction_xy = sheet_direction_3d / np.linalg.norm(sheet_direction_3d)
        
        # Determine edge direction (perpendicular to sheet direction for edge-to-edge contact)
        edge_direction = np.array([-sheet_direction_xy[1], sheet_direction_xy[0]])
        edge_direction = edge_direction / np.linalg.norm(edge_direction)
        
        return {
            'residues': sheet_residues,
            'direction': edge_direction,  # Direction perpendicular to beta-sheet (for edge contact)
            'sheet_direction': sheet_direction_xy,  # Direction along beta-sheet
            'center': np.mean([r['pos'][:2] for r in sheet_residues], axis=0)
        }
    
    def _fallback_beta_sheet_detection(self) -> Dict:
        """
        Fallback method: use CA atom distribution.
        
        Returns:
            Dict: Beta-sheet information
        """
        # Get CA atoms
        ca_atoms = self.monomer_atoms.select_atoms('name CA')
        ca_positions = ca_atoms.positions[:, :2]  # Only xy coordinates
        
        # PCA on CA positions
        pca = PCA(n_components=2)
        pca.fit(ca_positions)
        
        # Use second principal axis as potential sheet direction
        sheet_direction = pca.components_[1]
        edge_direction = np.array([-sheet_direction[1], sheet_direction[0]])
        edge_direction = edge_direction / np.linalg.norm(edge_direction)
        
        print(f"Fallback edge direction: [{edge_direction[0]:.3f}, {edge_direction[1]:.3f}]")
        
        return {
            'residues': [],
            'direction': edge_direction,
            'sheet_direction': sheet_direction,
            'center': np.mean(ca_positions, axis=0)
        }
    
    def build_dimer(self, 
                   separation_distance: float,
                   z_rotation_1: float = 0.0,
                   x_rotation_1: float = 0.0, 
                   y_rotation_1: float = 0.0,
                   z_rotation_2: float = 0.0,
                   x_rotation_2: float = 0.0,
                   y_rotation_2: float = 0.0,
                   output_pdb: Optional[str] = None) -> str:
        """
        Build a dimer with specified geometry parameters using asymmetric rotations.
        
        This positions monomers edge-to-edge based on detected beta-sheet regions,
        allowing different rotations for each monomer to explore more binding poses.
        
        Parameters:
            separation_distance (float): Distance between beta-sheet edges (Angstroms)
            z_rotation_1 (float): Z-rotation for monomer 1 (degrees)
            x_rotation_1 (float): X-rotation for monomer 1 (degrees)  
            y_rotation_1 (float): Y-rotation for monomer 1 (degrees)
            z_rotation_2 (float): Z-rotation for monomer 2 (degrees)
            x_rotation_2 (float): X-rotation for monomer 2 (degrees)
            y_rotation_2 (float): Y-rotation for monomer 2 (degrees)
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
        
        # Apply different rotations to each monomer
        protein1 = self._apply_rotations(protein1, z_rotation_1, x_rotation_1, y_rotation_1)
        protein2 = self._apply_rotations(protein2, z_rotation_2, x_rotation_2, y_rotation_2)
        
        # Position monomers edge-to-edge based on beta-sheet detection
        self._position_edge_to_edge(protein1, protein2, separation_distance)
        
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
    
    def _apply_rotations(self, protein_atoms, z_rot: float, x_rot: float, y_rot: float):
        """Apply rotations to protein atoms."""
        if z_rot != 0.0:
            protein_atoms = rotate.rotateby(
                angle=z_rot, direction=[0, 0, 1], ag=protein_atoms
            )(protein_atoms)
        
        if x_rot != 0.0:
            protein_atoms = rotate.rotateby(
                angle=x_rot, direction=[1, 0, 0], ag=protein_atoms
            )(protein_atoms)
        
        if y_rot != 0.0:
            protein_atoms = rotate.rotateby(
                angle=y_rot, direction=[0, 1, 0], ag=protein_atoms
            )(protein_atoms)
        
        return protein_atoms
    
    def _position_edge_to_edge(self, protein1, protein2, separation_distance: float):
        """
        Position two proteins edge-to-edge based on detected beta-sheet regions.
        """
        edge_direction = self.beta_sheet_info['direction']
        
        # Get backbone atoms for positioning
        backbone1 = protein1.select_atoms('backbone')
        backbone2 = protein2.select_atoms('backbone')
        
        # Project positions onto edge direction
        pos1 = backbone1.positions[:, :2]  # xy only
        pos2 = backbone2.positions[:, :2]
        
        proj1 = np.dot(pos1, edge_direction)
        proj2 = np.dot(pos2, edge_direction)
        
        # Position protein1 so its "far" edge is at -separation_distance/2
        far_edge_1 = proj1.max()
        target_pos_1 = -separation_distance/2
        translation_1 = np.append((target_pos_1 - far_edge_1) * edge_direction, 0)
        protein1.translate(translation_1)
        
        # Position protein2 so its "near" edge is at +separation_distance/2  
        near_edge_2 = proj2.min()
        target_pos_2 = separation_distance/2
        translation_2 = np.append((target_pos_2 - near_edge_2) * edge_direction, 0)
        protein2.translate(translation_2)
    
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
                         z_rotation_1: float = 0.0,
                         x_rotation_1: float = 0.0,
                         y_rotation_1: float = 0.0,
                         z_rotation_2: float = 0.0,
                         x_rotation_2: float = 0.0,
                         y_rotation_2: float = 0.0,
                         cleanup: bool = True) -> Dict[str, float]:
        """
        Build and score a dimer with given geometry parameters.
        
        Parameters:
            separation_distance (float): Distance between monomer centers
            z_rotation_1 (float): Z-rotation for monomer 1 (degrees)
            x_rotation_1 (float): X-rotation for monomer 1 (degrees)
            y_rotation_1 (float): Y-rotation for monomer 1 (degrees)
            z_rotation_2 (float): Z-rotation for monomer 2 (degrees)
            x_rotation_2 (float): X-rotation for monomer 2 (degrees)
            y_rotation_2 (float): Y-rotation for monomer 2 (degrees)
            cleanup (bool): Whether to delete temporary PDB file
            
        Returns:
            Dict[str, float]: Score dictionary with geometry parameters added
        """
        # Build dimer
        dimer_pdb = self.build_dimer(
            separation_distance=separation_distance,
            z_rotation_1=z_rotation_1,
            x_rotation_1=x_rotation_1,
            y_rotation_1=y_rotation_1,
            z_rotation_2=z_rotation_2,
            x_rotation_2=x_rotation_2,
            y_rotation_2=y_rotation_2
        )
        
        # Score dimer
        scores = self.score_dimer(dimer_pdb)
        
        # Add geometry parameters to results
        scores.update({
            'separation_distance': separation_distance,
            'z_rotation_1': z_rotation_1,
            'x_rotation_1': x_rotation_1,
            'y_rotation_1': y_rotation_1,
            'z_rotation_2': z_rotation_2,
            'x_rotation_2': x_rotation_2,
            'y_rotation_2': y_rotation_2
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
                         z_rotation_1: float = 0.0,
                         x_rotation_1: float = 0.0,
                         y_rotation_1: float = 0.0,
                         z_rotation_2: float = 0.0,
                         x_rotation_2: float = 0.0,
                         y_rotation_2: float = 0.0) -> Dict[str, float]:
        """
        Convert dimer geometry to ring building parameters.
        
        For rings, we'll use the average of the two monomer rotations as the
        standard rotation applied to all subunits.
        
        Parameters:
            separation_distance (float): Optimal separation from dimer optimization
            n_subunits (int): Number of subunits in target ring
            z_rotation_1, x_rotation_1, y_rotation_1: Rotations for monomer 1
            z_rotation_2, x_rotation_2, y_rotation_2: Rotations for monomer 2
            
        Returns:
            Dict[str, float]: Ring building parameters
        """
        radius = self.calculate_ring_radius(separation_distance, n_subunits)
        
        # Use average rotations for ring building
        avg_z_rotation = (z_rotation_1 + z_rotation_2) / 2
        avg_x_rotation = (x_rotation_1 + x_rotation_2) / 2
        avg_y_rotation = (y_rotation_1 + y_rotation_2) / 2
        
        return {
            'radius': radius,
            'z_rotation': avg_z_rotation,
            'x_rotation': avg_x_rotation,
            'y_rotation': avg_y_rotation,
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
    
    # Test a few different geometries with asymmetric rotations
    test_cases = [
        {'separation_distance': 10.0, 'z_rotation_1': 0.0, 'z_rotation_2': 0.0},
        {'separation_distance': 12.0, 'z_rotation_1': 0.0, 'z_rotation_2': 180.0},
        {'separation_distance': 15.0, 'z_rotation_1': 90.0, 'z_rotation_2': 270.0},
        {'separation_distance': 12.0, 'z_rotation_1': 0.0, 'z_rotation_2': 0.0, 'x_rotation_1': 10.0},
    ]
    
    print("\nTesting different dimer geometries with asymmetric rotations:")
    print("=" * 80)
    
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
        for n_subunits in [6, 8, 10, 12]:
            ring_params = builder.get_ring_geometry(
                scores['separation_distance'], 
                n_subunits,
                scores['z_rotation_1'], scores['x_rotation_1'], scores['y_rotation_1'],
                scores['z_rotation_2'], scores['x_rotation_2'], scores['y_rotation_2']
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