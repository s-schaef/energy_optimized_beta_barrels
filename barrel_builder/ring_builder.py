#!/usr/bin/env python3

import os
import argparse
import tempfile
import warnings
import numpy as np
import MDAnalysis as mda
from string import ascii_uppercase as auc
from string import ascii_lowercase as alc
from typing import Dict

# PyRosetta is imported lazily in _get_score_function() so that
# ring building and geometry tests can run without it installed.
pyrosetta = None
rosetta = None

# Empirical weights of the PyRosetta terms used to score assemblies.
# fa_atr and fa_rep are downscaled to tolerate minor overlaps of the rigid
# protomers; backbone hydrogen bonds (hbond_lr_bb) are upweighted.
SCORE_WEIGHTS = {
    'fa_atr': 0.9,
    'fa_rep': 0.02,
    'hbond_sr_bb': 1.0,
    'hbond_lr_bb': 10.0,
}

# One score function per process (PyRosetta is initialized only once).
_SCOREFXN = None

warnings.filterwarnings('ignore', message='.*CRYST1 record.*')
warnings.filterwarnings('ignore', message='.*Reader has no dt information.*')
warnings.filterwarnings('ignore', message='.*Using default value of.*')


def _get_score_function():
    """Initialize PyRosetta once per process and return the reweighted score function."""
    global pyrosetta, rosetta, _SCOREFXN
    if _SCOREFXN is not None:
        return _SCOREFXN

    import pyrosetta as _pyrosetta
    from pyrosetta import rosetta as _rosetta
    pyrosetta = _pyrosetta
    rosetta = _rosetta

    pyrosetta.init('-mute all')

    scorefxn = pyrosetta.create_score_function('empty')
    for term, weight in SCORE_WEIGHTS.items():
        scorefxn.set_weight(getattr(rosetta.core.scoring, term), weight)
    _SCOREFXN = scorefxn

    print("PyRosetta initialized with scoring terms: "
          + ", ".join(f"{t} (x{w})" for t, w in SCORE_WEIGHTS.items()))
    return _SCOREFXN


class RingBuilder:
    """
    Build circular protein assemblies from aligned monomer structures.

    The monomer should be pre-aligned (see barrel_builder.alignment) so that
    its beta-sheet runs along the z-axis and faces the ring center once the
    monomer is placed on the positive x-axis.
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

        print(f"Initialized RingBuilder with monomer containing "
              f"{len(self.monomer_atoms)} protein atoms")

    def _initialize_pyrosetta(self):
        """Initialize PyRosetta with the empirically reweighted scoring function."""
        if self.scorefxn is None:
            self.scorefxn = _get_score_function()

    def build_ring(self, n_subunits: int = 30, radius: float = 50.0,
                   tilt_angle: float = 0.0, cone_angle: float = 0.0):
        """
        Build a circular (C_n symmetric) assembly of rigid protein subunits.

        Each subunit is rotated about its own center of geometry (tilt around x,
        cone angle around y, then around z to face the ring axis) and
        translated to its position on the ring.

        Parameters:
            n_subunits (int): Number of subunits in the ring (2-52)
            radius (float): Distance in Angstroms from the ring axis to the
                center of geometry of each subunit (not the pore lumen radius)
            tilt_angle (float): Rotation of each subunit around the radial
                (x) axis in degrees
            cone_angle (float): Rotation of each subunit around the tangential
                (y) axis in degrees; positive values make the ring narrower
                at the bottom (-z)

        Returns:
            MDAnalysis.Universe: The assembled ring structure
        """
        if n_subunits < 2:
            raise ValueError("n_subunits must be at least 2")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if n_subunits > 52:
            raise ValueError("n_subunits cannot exceed 52 (limited by available segment IDs)")

        tmp_universes = []
        segid_list = auc + alc

        for idx in range(n_subunits):
            subunit = self.monomer_universe.copy()
            protein = subunit.select_atoms('protein')

            if tilt_angle != 0.0:
                protein.rotateby(tilt_angle, axis=[1, 0, 0])

            if cone_angle != 0.0:
                protein.rotateby(cone_angle, axis=[0, 1, 0])

            angle = 360 * idx / n_subunits
            protein.rotateby(angle, axis=[0, 0, 1])

            x_pos = radius * np.cos(np.radians(angle))
            y_pos = radius * np.sin(np.radians(angle))

            protein.translate([x_pos, y_pos, 0])

            protein.segments.segids = segid_list[idx]
            protein.atoms.chainIDs = segid_list[idx]
            tmp_universes.append(protein)

        self.ring = mda.Merge(*[u.atoms for u in tmp_universes])

        print(f"Built ring with {n_subunits} subunits, radius {radius:.1f} A, "
              f"tilt {tilt_angle:.1f} deg, cone {cone_angle:.1f} deg")
        return self.ring

    def center_ring(self):
        """Center the ring assembly at the origin."""
        if self.ring is None:
            raise RuntimeError("Ring must be built before centering. Call build_ring() first.")
        center_of_mass = self.ring.atoms.center_of_mass()
        self.ring.atoms.translate(-center_of_mass)

    def write_ring_pdb(self, output_pdb: str, centered: bool = True):
        """Write the assembled ring to a PDB file."""
        if self.ring is None:
            raise RuntimeError("Ring must be built before writing. Call build_ring() first.")
        if centered:
            self.center_ring()
        self.ring.atoms.write(output_pdb)
        print(f"Ring assembly written to {output_pdb}")

    def score_ring(self) -> Dict[str, float]:
        """
        Score the assembled ring using the reweighted PyRosetta score function.

        The rigid assembly is scored as is (no minimization or repacking).
        All returned components are weighted (see SCORE_WEIGHTS) and sum
        to total_score. Scores are only meaningful for comparing geometries
        built from the same monomer.
        """
        if self.ring is None:
            raise RuntimeError("Ring must be built before scoring. Call build_ring() first.")

        if self.scorefxn is None:
            self._initialize_pyrosetta()

        temporary = False
        if not self.output_pdb:
            temporary = True
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
                temp_pdb_path = tmp.name
            self.output_pdb = temp_pdb_path

        try:
            self.write_ring_pdb(self.output_pdb, centered=True)
            pose = pyrosetta.pose_from_pdb(self.output_pdb)

            scores = {'total_score': self.scorefxn(pose)}
            for term in SCORE_WEIGHTS:
                scores[term] = self.scorefxn.score_by_scoretype(
                    pose, getattr(rosetta.core.scoring, term))
            scores['n_subunits'] = len(self.ring.segments)
            return scores

        finally:
            if temporary and os.path.exists(temp_pdb_path):
                os.remove(temp_pdb_path)
                self.output_pdb = None

    def get_ring_info(self):
        """Get information about the current ring assembly."""
        if self.ring is None:
            return None
        return {
            'n_atoms': len(self.ring.atoms),
            'n_residues': len(self.ring.residues),
            'n_segments': len(self.ring.segments),
            'center_of_mass': self.ring.atoms.center_of_mass(),
            'dimensions': self.ring.dimensions
        }


def main():
    """CLI entry point for barrel-build."""
    parser = argparse.ArgumentParser(
        description='Build a circular protein assembly from an aligned monomer PDB file.')
    parser.add_argument('--input', required=True, help='Aligned monomer PDB file')
    parser.add_argument('--output', required=True, help='Output PDB file for the ring')
    parser.add_argument('--radius', type=float, default=120.0,
                        help='Distance from ring axis to subunit center of geometry '
                             'in Angstroms (default: 120.0)')
    parser.add_argument('--tilt_angle', type=float, default=-16.0,
                        help='Tilt angle around the x-axis in degrees (default: -16.0)')
    parser.add_argument('--n_subunits', type=int, default=30,
                        help='Number of subunits, 2-52 (default: 30)')
    parser.add_argument('--score', action='store_true',
                        help='Score the assembly with PyRosetta')
    parser.add_argument('--cone_angle', type=float, default=0.0,
                        help='Rotation around the tangential (y) axis in degrees; positive '
                             'values make the ring narrower at the bottom (default: 0.0)')

    args = parser.parse_args()

    try:
        builder = RingBuilder(args.input)
        builder.output_pdb = args.output

        builder.build_ring(args.n_subunits, args.radius, args.tilt_angle, args.cone_angle)

        if args.score:
            score = builder.score_ring()
            print(f"Total score: {score['total_score']:.2f}")
            for term, weight in SCORE_WEIGHTS.items():
                print(f"  {term}: {score[term]:.2f} (weight {weight})")
        else:
            builder.write_ring_pdb(args.output)

        info = builder.get_ring_info()
        if info:
            print(f"Ring info: {info['n_atoms']} atoms, {info['n_residues']} residues, "
                  f"{info['n_segments']} segments")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
