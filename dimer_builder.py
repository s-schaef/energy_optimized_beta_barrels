#!/usr/bin/env python3

import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import rotate
import pyrosetta
from pyrosetta import rosetta
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix, ConvexHull
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional
import os
import tempfile
import warnings

class DimerBuilder:
    """
    Build and score protein dimers with deterministic beta-sheet edge-to-edge positioning.
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
        
        # Detect beta-sheets in the monomer
        self.beta_sheet_info = self._detect_largest_beta_sheet()
        
        if self.beta_sheet_info is None:
            warnings.warn("No beta-sheet detected. Will use geometric approach for positioning.")
            self.has_beta_sheet = False
        else:
            self.has_beta_sheet = True
            print(f"Detected largest beta-sheet:")
            print(f"  {self.beta_sheet_info['num_residues']} residues")
            print(f"  Area: {self.beta_sheet_info['area']:.1f} Å²")
            print(f"  Long edge: {self.beta_sheet_info['rectangle']['long_edge_length']:.1f} Å")
            print(f"  Short edge: {self.beta_sheet_info['rectangle']['short_edge_length']:.1f} Å")
    
    def _initialize_pyrosetta(self):
        """Initialize PyRosetta with appropriate scoring function."""
        pyrosetta.init('-mute all')
        
        self.scorefxn = pyrosetta.create_score_function('empty')
        self.scorefxn.set_weight(rosetta.core.scoring.fa_atr, 1.0)
        self.scorefxn.set_weight(rosetta.core.scoring.fa_rep, 1.0)
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_bb_sc, 1.0)
        
        print("PyRosetta initialized with scoring terms: fa_atr, fa_rep, hbond_bb_sc")
    
    def _detect_largest_beta_sheet(self) -> Optional[Dict]:
        """
        Detect the largest beta-sheet in the monomer.
        
        Returns:
            Dict with beta-sheet information or None if detection fails
        """
        try:
            # Load structure in PyRosetta
            pose = pyrosetta.pose_from_pdb(self.monomer_pdb)
            
            # Assign secondary structure
            from pyrosetta.rosetta.core.scoring.dssp import Dssp
            dssp = Dssp(pose)
            dssp.insert_ss_into_pose(pose)
            
            # Get beta-strand residues
            beta_residues = []
            for i in range(1, pose.total_residue() + 1):
                if pose.secstruct(i) == 'E':
                    residue = pose.residue(i)
                    ca_atom = residue.atom("CA")
                    beta_residues.append({
                        'res_num': i,
                        'ca_pos': np.array([ca_atom.xyz().x, ca_atom.xyz().y, ca_atom.xyz().z])
                    })
            
            if len(beta_residues) < 4:
                return None
            
            # Group into sheets based on proximity
            sheets = self._group_into_sheets(beta_residues)
            
            # Find largest sheet by fitting rectangles
            largest_sheet = None
            max_area = 0
            
            for sheet_residues in sheets:
                sheet_info = self._characterize_sheet(sheet_residues)
                if sheet_info and sheet_info['area'] > max_area:
                    max_area = sheet_info['area']
                    largest_sheet = sheet_info
            
            return largest_sheet
            
        except Exception as e:
            print(f"Error in beta-sheet detection: {e}")
            return None
    
    def _group_into_sheets(self, beta_residues: List[Dict]) -> List[List[Dict]]:
        """Group beta-strand residues into connected sheets based on spatial proximity."""
        if not beta_residues:
            return []
        
        # Build distance matrix
        ca_positions = np.array([r['ca_pos'] for r in beta_residues])
        dist_matrix = distance_matrix(ca_positions, ca_positions)
        
        # Simple clustering based on proximity
        sheets = []
        unassigned = set(range(len(beta_residues)))
        
        while unassigned:
            current_sheet = [unassigned.pop()]
            
            changed = True
            while changed:
                changed = False
                for i in list(unassigned):
                    for j in current_sheet:
                        if dist_matrix[i, j] < 10.0:  # 10 Å threshold
                            current_sheet.append(i)
                            unassigned.remove(i)
                            changed = True
                            break
            
            sheet_residues = [beta_residues[i] for i in current_sheet]
            sheets.append(sheet_residues)
        
        return sheets
    
    def _characterize_sheet(self, sheet_residues: List[Dict]) -> Optional[Dict]:
        """Characterize a beta-sheet: fit plane and rectangle."""
        if len(sheet_residues) < 4:
            return None
        
        ca_positions = np.array([r['ca_pos'] for r in sheet_residues])
        
        # Fit plane using PCA
        center = np.mean(ca_positions, axis=0)
        centered = ca_positions - center
        pca = PCA(n_components=3)
        pca.fit(centered)
        normal = pca.components_[2]  # Normal is the smallest variance component
        
        # Project points to plane and fit rectangle
        # Create coordinate system on plane
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Project to 2D
        points_2d = []
        for pos in ca_positions:
            p = pos - center
            x = np.dot(p, u)
            y = np.dot(p, v)
            points_2d.append([x, y])
        points_2d = np.array(points_2d)
        
        # Find minimum area rectangle
        try:
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
        except:
            hull_points = points_2d
        
        min_area = float('inf')
        best_rect = None
        
        for i in range(len(hull_points)):
            edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
            angle = np.arctan2(edge[1], edge[0])
            
            # Rotate points
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = np.dot(points_2d, rot_matrix.T)
            
            # Bounding box
            min_x, min_y = rotated.min(axis=0)
            max_x, max_y = rotated.max(axis=0)
            width = max_x - min_x
            height = max_y - min_y
            area = width * height
            
            if area < min_area:
                min_area = area
                
                # Rectangle corners in 2D
                corners_2d = np.array([
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y]
                ])
                
                # Rotate back
                inv_rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
                corners_2d = np.dot(corners_2d, inv_rot.T)
                
                # Convert to 3D
                corners_3d = []
                for c2d in corners_2d:
                    c3d = center + c2d[0] * u + c2d[1] * v
                    corners_3d.append(c3d)
                
                best_rect = {
                    'corners': np.array(corners_3d),
                    'width': width,
                    'height': height,
                    'area': area,
                    'long_edge_length': max(width, height),
                    'short_edge_length': min(width, height),
                    'normal': normal,
                    'center': center,
                    'u_axis': u,
                    'v_axis': v
                }
        
        if best_rect:
            # Identify which edges are long and short
            corners = best_rect['corners']
            edge_vectors = [
                corners[1] - corners[0],
                corners[2] - corners[1],
                corners[3] - corners[2],
                corners[0] - corners[3]
            ]
            edge_lengths = [np.linalg.norm(v) for v in edge_vectors]
            
            # Find long and short edges
            if edge_lengths[0] > edge_lengths[1]:
                # Edges 0-1 and 2-3 are long
                best_rect['long_edge_indices'] = [(0, 1), (2, 3)]
                best_rect['short_edge_indices'] = [(1, 2), (3, 0)]
                best_rect['long_edge_direction'] = edge_vectors[0] / edge_lengths[0]
                best_rect['short_edge_direction'] = edge_vectors[1] / edge_lengths[1]
            else:
                # Edges 1-2 and 3-0 are long
                best_rect['long_edge_indices'] = [(1, 2), (3, 0)]
                best_rect['short_edge_indices'] = [(0, 1), (2, 3)]
                best_rect['long_edge_direction'] = edge_vectors[1] / edge_lengths[1]
                best_rect['short_edge_direction'] = edge_vectors[0] / edge_lengths[0]
            
            return {
                'residues': sheet_residues,
                'num_residues': len(sheet_residues),
                'rectangle': best_rect,
                'area': area
            }
        
        return None
    
    def calculate_deterministic_position(self, n_subunits: int) -> Dict[str, float]:
        """
        Calculate the deterministic position for beta-sheet edge-to-edge connection.
        
        Parameters:
            n_subunits (int): Number of subunits in the target ring
            
        Returns:
            Dict with positioning parameters
        """
        if not self.has_beta_sheet:
            # Fallback for no beta-sheet
            return {
                'use_deterministic': False,
                'separation_distance': 20.0,  # Default
                'rotation_angle': 360.0 / n_subunits
            }
        
        rect = self.beta_sheet_info['rectangle']
        
        # The separation should be along the short edge direction
        # Distance should be approximately the protein diameter
        protein_diameter = self._estimate_protein_diameter()
        
        # The rotation angle for circular assembly
        rotation_angle = 360.0 / n_subunits
        
        return {
            'use_deterministic': True,
            'separation_distance': protein_diameter,
            'rotation_angle': rotation_angle,
            'short_edge_direction': rect['short_edge_direction'],
            'long_edge_direction': rect['long_edge_direction'],
            'sheet_normal': rect['normal'],
            'sheet_center': rect['center']
        }
    
    def _estimate_protein_diameter(self) -> float:
        """Estimate protein diameter from all atoms."""
        positions = self.monomer_atoms.positions
        # Use convex hull for better estimate
        try:
            hull = ConvexHull(positions[:, :2])  # xy projection
            hull_points = positions[hull.vertices, :2]
            # Maximum distance between hull points
            max_dist = 0
            for i in range(len(hull_points)):
                for j in range(i+1, len(hull_points)):
                    dist = np.linalg.norm(hull_points[i] - hull_points[j])
                    max_dist = max(max_dist, dist)
            return max_dist
        except:
            # Fallback to simple range
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            return max(x_range, y_range)
    
    def build_dimer_deterministic(self,
                                 n_subunits: int = 12,
                                 separation_adjustment: float = 0.0,
                                 angle_adjustment: float = 0.0,
                                 output_pdb: Optional[str] = None) -> str:
        """
        Build dimer using deterministic beta-sheet positioning.
        
        Parameters:
            n_subunits (int): Target number of subunits for ring
            separation_adjustment (float): Adjustment to separation distance (Å)
            angle_adjustment (float): Adjustment to rotation angle (degrees)
            output_pdb (str): Output PDB filename
            
        Returns:
            str: Path to dimer PDB file
        """
        if not self.has_beta_sheet:
            raise ValueError("Deterministic positioning requires beta-sheet detection")
        
        # Get deterministic parameters
        det_params = self.calculate_deterministic_position(n_subunits)
        
        # Create two copies of the monomer
        monomer1 = self.monomer_universe.copy()
        monomer2 = self.monomer_universe.copy()
        
        protein1 = monomer1.select_atoms('protein')
        protein2 = monomer2.select_atoms('protein')
        
        # Position using deterministic approach
        rect = self.beta_sheet_info['rectangle']
        
        # Step 1: Orient protein1 with beta-sheet in standard position
        # Center at origin
        center1 = protein1.center_of_mass()
        protein1.translate(-center1)
        
        # We want the long edge along x-axis and short edge along y-axis
        # This is already roughly the case if the protein was pre-aligned
        
        # Step 2: Position protein2
        # First, copy protein1's position
        center2 = protein2.center_of_mass()
        protein2.translate(-center2)
        
        # Move along the short edge direction (y-axis if properly oriented)
        separation = det_params['separation_distance'] + separation_adjustment
        protein2.translate([0, separation, 0])
        
        # Rotate for circular assembly
        # The rotation should be around the connection axis (x-axis for long edge connection)
        angle = det_params['rotation_angle'] + angle_adjustment
        
        # Apply half angle to each protein for symmetry
        protein1 = rotate.rotateby(angle=-angle/2, direction=[1, 0, 0], ag=protein1)(protein1)
        protein2 = rotate.rotateby(angle=angle/2, direction=[1, 0, 0], ag=protein2)(protein2)
        
        # Also rotate protein2 by 180° around z-axis so opposite long edges face each other
        protein2 = rotate.rotateby(angle=180, direction=[0, 0, 1], ag=protein2)(protein2)
        
        # Merge the two monomers
        dimer = mda.Merge(protein1, protein2)
        
        # Update segment IDs
        dimer.segments[0].segid = 'A'
        dimer.segments[1].segid = 'B'
        
        # Write dimer to file
        if output_pdb is None:
            temp_fd, output_pdb = tempfile.mkstemp(suffix='.pdb', prefix='dimer_')
            os.close(temp_fd)
        
        dimer.select_atoms('all').write(output_pdb)
        
        return output_pdb
    
    def build_dimer(self, 
                   separation_distance: float,
                   z_rotation_1: float = 0.0,
                   x_rotation_1: float = 0.0, 
                   y_rotation_1: float = 0.0,
                   z_rotation_2: float = 0.0,
                   x_rotation_2: float = 0.0,
                   y_rotation_2: float = 0.0,
                   n_subunits: int = 12,
                   output_pdb: Optional[str] = None) -> str:
        """
        Build a dimer with specified rotations and separation.
        This is the original method for compatibility with the optimizer.
        
        For beta-sheet proteins, consider using build_dimer_deterministic instead.
        """
        # Create two copies of the monomer
        monomer1 = self.monomer_universe.copy()
        monomer2 = self.monomer_universe.copy()
        
        protein1 = monomer1.select_atoms('protein')
        protein2 = monomer2.select_atoms('protein')
        
        # Apply individual rotations
        protein1 = self._apply_rotations(protein1, z_rotation_1, x_rotation_1, y_rotation_1)
        protein2 = self._apply_rotations(protein2, z_rotation_2, x_rotation_2, y_rotation_2)
        
        # Position proteins
        if self.has_beta_sheet:
            # Use beta-sheet aware positioning
            protein1, protein2 = self._position_with_beta_sheets(
                protein1, protein2, separation_distance, n_subunits
            )
        else:
            # Fallback to geometric positioning
            protein1, protein2 = self._position_geometric(
                protein1, protein2, separation_distance, n_subunits
            )
        
        # Merge the two monomers
        dimer = mda.Merge(protein1, protein2)
        
        # Update segment IDs
        dimer.segments[0].segid = 'A'
        dimer.segments[1].segid = 'B'
        
        # Write dimer to file
        if output_pdb is None:
            temp_fd, output_pdb = tempfile.mkstemp(suffix='.pdb', prefix='dimer_')
            os.close(temp_fd)
        
        dimer.select_atoms('all').write(output_pdb)
        
        return output_pdb
    
    def _apply_rotations(self, protein_atoms, z_rot: float, x_rot: float, y_rot: float):
        """Apply rotations to protein atoms."""
        if z_rot != 0.0:
            protein_atoms = rotate.rotateby(angle=z_rot, direction=[0, 0, 1], ag=protein_atoms)(protein_atoms)
        if x_rot != 0.0:
            protein_atoms = rotate.rotateby(angle=x_rot, direction=[1, 0, 0], ag=protein_atoms)(protein_atoms)
        if y_rot != 0.0:
            protein_atoms = rotate.rotateby(angle=y_rot, direction=[0, 1, 0], ag=protein_atoms)(protein_atoms)
        return protein_atoms
    
    def _position_with_beta_sheets(self, protein1, protein2, separation_distance: float, n_subunits: int):
        """Position proteins using detected beta-sheet information."""
        # Center both proteins
        center1 = protein1.center_of_mass()
        protein1.translate(-center1)
        
        center2 = protein2.center_of_mass()
        protein2.translate(-center2)
        
        # Rotate protein2 by 180° so opposite edges face
        protein2 = rotate.rotateby(angle=180, direction=[0, 0, 1], ag=protein2)(protein2)
        
        # Separate along y-axis (short edge direction)
        protein2.translate([0, separation_distance, 0])
        
        # Add wedge angle for circular assembly
        angle_between_subunits = 360.0 / n_subunits
        protein1 = rotate.rotateby(angle=-angle_between_subunits/2, direction=[1, 0, 0], ag=protein1)(protein1)
        protein2 = rotate.rotateby(angle=angle_between_subunits/2, direction=[1, 0, 0], ag=protein2)(protein2)
        
        return protein1, protein2
    
    def _position_geometric(self, protein1, protein2, separation_distance: float, n_subunits: int):
        """Fallback geometric positioning."""
        # Center both proteins
        center1 = protein1.center_of_mass()
        protein1.translate(-center1)
        
        center2 = protein2.center_of_mass()
        protein2.translate(-center2)
        
        # Rotate protein2 by 180°
        protein2 = rotate.rotateby(angle=180, direction=[0, 0, 1], ag=protein2)(protein2)
        
        # Separate along x-axis
        protein2.translate([separation_distance, 0, 0])
        
        # Add wedge angle
        angle_between_subunits = 360.0 / n_subunits
        protein1 = rotate.rotateby(angle=-angle_between_subunits/2, direction=[0, 1, 0], ag=protein1)(protein1)
        protein2 = rotate.rotateby(angle=angle_between_subunits/2, direction=[0, 1, 0], ag=protein2)(protein2)
        
        return protein1, protein2
    
    def score_dimer(self, dimer_pdb: str) -> Dict[str, float]:
        """Score a dimer using PyRosetta."""
        pose = pyrosetta.pose_from_pdb(dimer_pdb)
        total_score = self.scorefxn(pose)
        
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
                         n_subunits: int = 12,
                         cleanup: bool = True) -> Dict[str, float]:
        """Build and score a dimer with given geometry parameters."""
        dimer_pdb = self.build_dimer(
            separation_distance=separation_distance,
            z_rotation_1=z_rotation_1,
            x_rotation_1=x_rotation_1,
            y_rotation_1=y_rotation_1,
            z_rotation_2=z_rotation_2,
            x_rotation_2=x_rotation_2,
            y_rotation_2=y_rotation_2,
            n_subunits=n_subunits
        )
        
        scores = self.score_dimer(dimer_pdb)
        
        scores.update({
            'separation_distance': separation_distance,
            'z_rotation_1': z_rotation_1,
            'x_rotation_1': x_rotation_1,
            'y_rotation_1': y_rotation_1,
            'z_rotation_2': z_rotation_2,
            'x_rotation_2': x_rotation_2,
            'y_rotation_2': y_rotation_2,
            'n_subunits': n_subunits
        })
        
        if cleanup and os.path.exists(dimer_pdb):
            os.remove(dimer_pdb)
        
        return scores
    
    def evaluate_deterministic(self,
                             n_subunits: int = 12,
                             separation_adjustment: float = 0.0,
                             angle_adjustment: float = 0.0,
                             cleanup: bool = True) -> Dict[str, float]:
        """
        Evaluate the deterministic positioning with small adjustments.
        
        Parameters:
            n_subunits (int): Target number of subunits
            separation_adjustment (float): Adjustment to calculated separation (Å)
            angle_adjustment (float): Adjustment to calculated angle (degrees)
            cleanup (bool): Whether to delete temporary files
            
        Returns:
            Dict with scores and parameters
        """
        if not self.has_beta_sheet:
            raise ValueError("Deterministic evaluation requires beta-sheet detection")
        
        # Build dimer
        dimer_pdb = self.build_dimer_deterministic(
            n_subunits=n_subunits,
            separation_adjustment=separation_adjustment,
            angle_adjustment=angle_adjustment
        )
        
        # Score it
        scores = self.score_dimer(dimer_pdb)
        
        # Add parameters
        det_params = self.calculate_deterministic_position(n_subunits)
        scores.update({
            'n_subunits': n_subunits,
            'base_separation': det_params['separation_distance'],
            'base_angle': det_params['rotation_angle'],
            'separation_adjustment': separation_adjustment,
            'angle_adjustment': angle_adjustment,
            'final_separation': det_params['separation_distance'] + separation_adjustment,
            'final_angle': det_params['rotation_angle'] + angle_adjustment
        })
        
        if cleanup and os.path.exists(dimer_pdb):
            os.remove(dimer_pdb)
        
        return scores
    
    def calculate_ring_radius(self, separation_distance: float, n_subunits: int) -> float:
        """Calculate ring radius from separation distance."""
        angle_between_subunits = 2 * np.pi / n_subunits
        radius = separation_distance / (2 * np.sin(angle_between_subunits / 2))
        return radius


def test_deterministic_positioning(monomer_pdb: str):
    """Test the deterministic positioning approach."""
    print("Testing deterministic beta-sheet positioning...")
    
    builder = DimerBuilder(monomer_pdb)
    
    if not builder.has_beta_sheet:
        print("No beta-sheet detected - cannot test deterministic positioning")
        return
    
    # Test for different ring sizes
    for n_subunits in [6, 8, 10, 12]:
        print(f"\n{n_subunits}-mer ring:")
        
        # Get deterministic parameters
        det_params = builder.calculate_deterministic_position(n_subunits)
        print(f"  Base separation: {det_params['separation_distance']:.1f} Å")
        print(f"  Base angle: {det_params['rotation_angle']:.1f}°")
        
        # Test with no adjustment
        scores = builder.evaluate_deterministic(n_subunits=n_subunits)
        print(f"  No adjustment - Score: {scores['total_score']:.2f}")
        
        # Test small adjustments
        for sep_adj in [-2, 0, 2]:
            for ang_adj in [-5, 0, 5]:
                if sep_adj == 0 and ang_adj == 0:
                    continue  # Already tested
                
                scores = builder.evaluate_deterministic(
                    n_subunits=n_subunits,
                    separation_adjustment=sep_adj,
                    angle_adjustment=ang_adj
                )
                print(f"  Sep adj: {sep_adj:+.1f}, Ang adj: {ang_adj:+.1f} - Score: {scores['total_score']:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build dimers with deterministic beta-sheet positioning')
    parser.add_argument('--monomer', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--test', action='store_true', help='Run deterministic positioning test')
    
    args = parser.parse_args()
    
    if args.test:
        test_deterministic_positioning(args.monomer)