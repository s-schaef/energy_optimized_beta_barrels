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
import shutil
import tempfile


class RingBuilder:
    """
    Build circular protein assemblies. 
    """
    
    def __init__(self, monomer_pdb: str, tilt_angle: float = 0.0, radius: float = 50.0):
        """
        Initialize ring builder with aligned monomer.
        
        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
        """
        self.monomer_pdb = monomer_pdb
        self.monomer_universe = mda.Universe(monomer_pdb)
        self.monomer_atoms = self.monomer_universe.select_atoms('protein')
        self.tilt_angle = tilt_angle
        self.radius = radius
        

    def _initialize_pyrosetta(self):
        """Initialize PyRosetta with appropriate scoring function."""
        pyrosetta.init('-mute all')
        
        self.scorefxn = pyrosetta.create_score_function('empty')
        self.scorefxn.set_weight(rosetta.core.scoring.fa_atr, 1.0)
        self.scorefxn.set_weight(rosetta.core.scoring.fa_rep, 1.0)
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_bb_sc, 1.0)
        
        print("PyRosetta initialized with scoring terms: fa_atr, fa_rep, hbond_bb_sc")

    def build_ring(self, n_subunits, radius):
        """        Build a circular assembly of protein subunits.       

        Parameters:
            n_subunits (int): Number of subunits in the ring
            radius (float): Radius of the circular assembly
        """
        tmp_universes = []
        segid_list = (auc + alc)  # Create a list of segment IDs (A-Z, a-z)

        for idx in range(n_subunits):
            subunit = self.monomer_universe.copy()
            protein = subunit.select_atoms('protein')
            # first rotate the subunit around the y-axis
            # for the aligned monomer this is the tilt angle of the beta-sheet
            protein.rotateby(self.tilt_angle, axis=[0, 1, 0],)


            # move subunit to its position in the ring
            angle = 360/n_subunits * idx
            angle *= np.pi/180
            protein.translate([radius*np.cos(angle), radius*np.sin(angle), 0])
            

            # add subunit to the list of universes
            protein.segments.segids = segid_list[idx]
            tmp_universes.append(protein)

        # Combine all subunits into a single universe
        # Merge all universes into one
        ring = mda.Merge(*[u.atoms for u in tmp_universes])

        return ring
    


