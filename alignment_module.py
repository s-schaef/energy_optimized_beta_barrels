#!/usr/bin/env python3


import warnings
import numpy as np
import MDAnalysis as mda
#from MDAnalysis.analysis import align
#from MDAnalysis.transformations import rotate
from MDAnalysis.analysis.dssp import DSSP
from sklearn.decomposition import PCA
#from scipy.spatial import distance_matrix, ConvexHull
#from typing import Dict, List, Tuple, Optional

def are_strands_connected(universe, strand1, strand2, distance_cutoff=3.5, min_hbonds=2):
        """
        Check if two beta strands are connected by hydrogen bonds.
        
        Args:
            strand1: Tuple (start_resid, end_resid) for first strand
            strand2: Tuple (start_resid, end_resid) for second strand
            distance_cutoff: Maximum distance for H-bond (Angstroms)
            min_hbonds: Minimum number of H-bonds to consider strands connected
            
        Returns:
            bool: True if strands are connected
        """
        # Get backbone atoms for each strand
        strand1_atoms = universe.select_atoms(f"backbone and resid {strand1[0]}:{strand1[1]}")
        strand2_atoms = universe.select_atoms(f"backbone and resid {strand2[0]}:{strand2[1]}")
        
        # Get O and N atoms
        strand1_O = strand1_atoms.select_atoms("name O")
        strand1_N = strand1_atoms.select_atoms("name N")
        strand2_O = strand2_atoms.select_atoms("name O")
        strand2_N = strand2_atoms.select_atoms("name N")
        
        bonds_found = 0
        
        # Check O1...N2 bonds
        for o1 in strand1_O:
            for n2 in strand2_N:
                dist = np.linalg.norm(o1.position - n2.position)
                if dist < distance_cutoff:
                    bonds_found += 1
        
        # Check O2...N1 bonds
        for o2 in strand2_O:
            for n1 in strand1_N:
                dist = np.linalg.norm(o2.position - n1.position)
                if dist < distance_cutoff:
                    bonds_found += 1
        
        return bonds_found >= min_hbonds
    
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
        self.pdb_file = pdb_file
        self.universe = mda.Universe(self.pdb_file)
        self.protein = self.universe.select_atoms('protein')

        # Detect beta-sheets in the monomer
        self.ss_dict = self.get_secondary_structure()

        # Identify beta strands
        self.beta_strands = self.identify_beta_strands()
        print(f"Found {len(self.beta_strands)} beta strands")
        if len(self.beta_strands) == 0:
            warnings.warn("No beta strands detected. Will use geometric approach for positioning.")
            self.has_beta_sheet = False
        else:
            self.has_beta_sheet = True
            print("Grouping strands into beta sheets...")
            self.beta_sheets = self.find_beta_sheets()
            print(f"Found {len(self.beta_sheets)} beta sheets")
            # find largest sheet
            self.largest_sheet_info = self.find_largest_sheet()
        
            
    def get_secondary_structure(self):
        """
        Extract secondary structure information using DSSP through MDAnalysis.

        Returns:
            dict: Residue ID to secondary structure mapping
            Universe: MDAnalysis Universe object
        """
        
        # Run DSSP analysis
        dssp = DSSP(self.universe).run()
        
        # Get secondary structure for each residue
        ss_dict = {}
        for i, (resid, ss) in enumerate(zip(self.universe.residues.resids, dssp.results.dssp[0])):
            ss_dict[resid] = ss
        
        return ss_dict
    
    def identify_beta_strands(self):
        """
        Identify continuous beta strand segments.
        
        Args:
            ss_dict: Dictionary mapping residue IDs to secondary structure
            
        Returns:
            list: List of beta strand segments (start_resid, end_resid)
        """
        beta_strands = []
        current_strand = []
        
        sorted_resids = sorted(self.ss_dict.keys())
        
        for resid in sorted_resids:
            ss = self.ss_dict[resid]
            # 'E' represents extended strand (beta strand) in DSSP
            if ss == 'E':
                current_strand.append(resid)
            else:
                if current_strand:
                    beta_strands.append((current_strand[0], current_strand[-1]))
                    current_strand = []
        
        # Don't forget the last strand if it extends to the end
        if current_strand:
            beta_strands.append((current_strand[0], current_strand[-1]))
        
        return beta_strands
    
    
    def find_beta_sheets(self):
        """
        Group beta strands into sheets based on hydrogen bonding.
        
        Args:
            beta_strands: List of beta strand segments
            
        Returns:
            list: List of beta sheets, where each sheet is a list of strand indices
        """
        n_strands = len(self.beta_strands)
        sheets = []
        assigned = [False] * n_strands
        
        for i in range(n_strands):
            if assigned[i]:
                continue
                
            # Start a new sheet with this strand
            current_sheet = [i]
            assigned[i] = True
            
            # Keep looking for connected strands
            changed = True
            while changed:
                changed = False
                for j in range(n_strands):
                    if assigned[j]:
                        continue
                    
                    # Check if strand j is connected to any strand in current sheet
                    for strand_idx in current_sheet:
                        if are_strands_connected(self.universe, self.beta_strands[strand_idx], self.beta_strands[j]):
                            current_sheet.append(j)
                            assigned[j] = True
                            changed = True
                            break
            
            sheets.append(current_sheet)
        
        return sheets
    
    def find_largest_sheet(self):
        largest_sheet = None
        max_residues = 0
        
        for sheet in self.beta_sheets:
            # Calculate total residues in this sheet
            total_residues = sum(self.beta_strands[i][1] - self.beta_strands[i][0] + 1 for i in sheet)
            if total_residues > max_residues:
                max_residues = total_residues
                largest_sheet = sheet
        
        if largest_sheet:
            strands = [self.beta_strands[i] for i in largest_sheet]
            sheet_resids = []
            for strand in strands:
                sheet_resids.extend(range(strand[0], strand[1] + 1))

            result = {
                'strands': strands,
                'n_strands': len(largest_sheet),
                'n_residues': max_residues,
                'sheet_resids': sheet_resids,
                }
            
            print("\nLargest Beta Sheet Found:")
            print(f"  Number of strands: {result['n_strands']}")
            print(f"  Total residues: {result['n_residues']}")
            print("  Strands (start-end):")
            for strand in result['strands']:
                print(f"    {strand[0]}-{strand[1]}")
            
            return result
        else:
            print("No beta sheets found.")
            return None
    
        
    def align_to_standard_orientation(self, output_file=None):
        """
        Align monomer to standard orientation:
        1. Centers protein at origin (0,0,0)
        2. Align principal axis with z-axis
        3. Align second principal axis with x-axis
        
        Parameters:
            output_file (str, optional): If provided, write aligned structure to this file
            
        Returns:
            MDAnalysis.AtomGroup: Aligned protein atoms
        """
        if self.has_beta_sheet:
            # Align to beta-sheet standard orientation
            print("Aligning monomer using beta-sheet information.")
            
            sheet_resids = self.largest_sheet_info['sheet_resids']
            # Select atoms in the largest beta-sheet
            sheet_atoms = self.protein.select_atoms(f'resid {" ".join(map(str, sheet_resids))} and backbone')
            pca_positions = sheet_atoms.positions

        else:
            # Fallback to geometric positioning
            print("Aligning monomer using overall protein information.")
            # Step 1: Center the protein at origin
            center_of_mass = self.protein.center_of_mass()
            self.protein.translate(-center_of_mass)
        
            # Calculate principal axes using PCA on CA atoms
            ca_atoms = self.protein.select_atoms('name CA')
            if len(ca_atoms) == 0:
                raise ValueError("No CA atoms found in protein structure")
                
            pca_positions = ca_atoms.positions
            
        # Perform PCA to find principal axes
        pca = PCA(n_components=3)
        pca.fit(pca_positions)
        
        # Get the principal axes (eigenvectors)
        principal_axes = pca.components_
    
        z_axis = np.array([0, 0, 1])
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        # Source basis (principal components)
        source_basis = np.column_stack([principal_axes[0], principal_axes[1], principal_axes[2]])
        
        # Target basis
        target_basis = np.column_stack([z_axis, x_axis, y_axis])
        
        # Rotation matrix
        rotation_matrix = target_basis @ source_basis.T
        
        # Apply rotation to align principal axis with z-axis and second axis with x-axis
        self.protein.positions = (rotation_matrix @ self.protein.positions.T).T


        # Apply translation to center at origin
        if self.has_beta_sheet:
            # Center the beta-sheet atoms at origin
            center_of_mass = sheet_atoms.center_of_mass()
            self.protein.translate(-center_of_mass)
        else:
            # Center the entire protein at origin
            center_of_mass = self.protein.center_of_mass()
            self.protein.translate(-center_of_mass)

        
        # Write output if requested
        if output_file:
            self.protein.write(output_file)
            print(f"Aligned monomer written to {output_file}")
            
        return self.protein
    
    def get_aligned_monomer(self):
        """
        Get the aligned monomer atom group.
        
        Returns:
            MDAnalysis.AtomGroup: Aligned protein atoms
        """
        return self.protein
    


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
    # Position proteins
    aligner.align_to_standard_orientation(output_pdb)

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