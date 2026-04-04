#!/usr/bin/env python3

import os
import argparse
import warnings
import numpy as np
import MDAnalysis as mda
from string import ascii_uppercase as auc
from string import ascii_lowercase as alc
from typing import Dict

# PyRosetta is imported lazily in _initialize_pyrosetta() so that
# ring building and geometry tests can run without it installed.
pyrosetta = None
rosetta = None

warnings.filterwarnings('ignore', message='.*CRYST1 record.*')
warnings.filterwarnings('ignore', message='.*Reader has no dt information.*')
warnings.filterwarnings('ignore', message='.*Using default value of.*')


class RingBuilder:
    """
    Build circular protein assemblies from aligned monomer structures.

    The monomer should be pre-aligned so that the desired interface faces
    outward when positioned in the ring.
    """

    def __init__(self, monomer_pdb: str, gasdermin: bool = False):
        """
        Initialize ring builder with aligned monomer.

        Parameters:
            monomer_pdb (str): Path to aligned monomer PDB file
            gasdermin (bool): Enable gasdermin-specific y-axis rotation

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

        self.gasdermin = gasdermin
        self.ring = None
        self.scorefxn = None
        self.output_pdb = None

        print(f"Initialized RingBuilder with monomer containing "
              f"{len(self.monomer_atoms)} protein atoms")

    def _initialize_pyrosetta(self):
        """Initialize PyRosetta with a scoring function tuned for beta-barrel pore quality."""
        if self.scorefxn is not None:
            return

        global pyrosetta, rosetta
        import pyrosetta as _pyrosetta
        from pyrosetta import rosetta as _rosetta
        pyrosetta = _pyrosetta
        rosetta = _rosetta

        pyrosetta.init('-mute all')

        self.scorefxn = pyrosetta.create_score_function('empty')
        self.scorefxn.set_weight(rosetta.core.scoring.fa_atr, 0.9)
        self.scorefxn.set_weight(rosetta.core.scoring.fa_rep, 0.02)
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_sr_bb, 1.0)
        self.scorefxn.set_weight(rosetta.core.scoring.hbond_lr_bb, 10.0)

        print("PyRosetta initialized with scoring terms: "
              "fa_atr, fa_rep, hbond_sr_bb, hbond_lr_bb")

    def build_ring(self, n_subunits: int = 30, radius: float = 50.0,
                   tilt_angle: float = 0.0, z_offset: float = 0.0):
        """
        Build a circular assembly of protein subunits.

        Parameters:
            n_subunits (int): Number of subunits in the ring (2-52)
            radius (float): Radius of the circular assembly in Angstroms
            tilt_angle (float): Tilt angle of each subunit around x-axis in degrees
            z_offset (float): Alternating vertical offset between adjacent subunits
                in Angstroms. In real beta barrels, adjacent strands are staggered
                along the barrel axis to allow proper hydrogen bonding.

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

            if self.gasdermin:
                protein.rotateby(10, axis=[0, 1, 0])

            angle = 360 * idx / n_subunits
            protein.rotateby(angle, axis=[0, 0, 1])

            x_pos = radius * np.cos(np.radians(angle))
            y_pos = radius * np.sin(np.radians(angle))
            z_pos = z_offset * ((-1) ** idx)

            protein.translate([x_pos, y_pos, z_pos])

            protein.segments.segids = segid_list[idx]
            protein.atoms.chainIDs = segid_list[idx]
            tmp_universes.append(protein)

        self.ring = mda.Merge(*[u.atoms for u in tmp_universes])

        print(f"Built ring with {n_subunits} subunits, radius {radius:.1f} A, "
              f"tilt {tilt_angle:.1f} deg, z_offset {z_offset:.1f} A")
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
        Score the assembled ring using PyRosetta.

        Returns a dictionary with individual energy components and a composite
        pore_quality score that emphasizes inter-subunit hydrogen bonding
        while penalizing atomic clashes.
        """
        if self.ring is None:
            raise RuntimeError("Ring must be built before scoring. Call build_ring() first.")

        if self.scorefxn is None:
            self._initialize_pyrosetta()

        temporary = False
        if not self.output_pdb:
            temporary = True
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
                temp_pdb_path = tmp.name
            self.output_pdb = temp_pdb_path

        try:
            self.write_ring_pdb(self.output_pdb, centered=True)
            pose = pyrosetta.pose_from_pdb(self.output_pdb)

            total_score = self.scorefxn(pose)

            fa_atr = self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.fa_atr)
            fa_rep = self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.fa_rep)
            hbond_sr_bb = self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.hbond_sr_bb)
            hbond_lr_bb = self.scorefxn.score_by_scoretype(pose, rosetta.core.scoring.hbond_lr_bb)

            n_subunits = len(self.ring.segments)
            hbond_per_subunit = hbond_lr_bb / n_subunits if n_subunits > 0 else 0

            pore_quality = (
                hbond_lr_bb * 10.0
                + hbond_sr_bb * 1.0
                + fa_atr * 0.5
                + fa_rep * 0.5
            )

            return {
                'total_score': total_score,
                'fa_atr': fa_atr,
                'fa_rep': fa_rep,
                'hbond_sr_bb': hbond_sr_bb,
                'hbond_lr_bb': hbond_lr_bb,
                'hbond_per_subunit': hbond_per_subunit,
                'pore_quality': pore_quality,
                'n_subunits': n_subunits,
            }

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
    parser.add_argument('--input', required=True, help='Monomeric input PDB file')
    parser.add_argument('--output', required=True, help='Circular output PDB file')
    parser.add_argument('--radius', type=float, default=120.0,
                        help='Radius in Angstroms (default: 120.0)')
    parser.add_argument('--tilt_angle', type=float, default=-16.0,
                        help='Tilt angle in degrees (default: -16.0)')
    parser.add_argument('--z_offset', type=float, default=0.0,
                        help='Alternating z-offset in Angstroms (default: 0.0)')
    parser.add_argument('--n_subunits', type=int, default=30,
                        help='Number of subunits (default: 30)')
    parser.add_argument('--score', action='store_true',
                        help='Score the assembly with PyRosetta')
    parser.add_argument('--gasdermin', action='store_true',
                        help='Enable gasdermin-specific modifications')

    args = parser.parse_args()

    try:
        builder = RingBuilder(args.input, gasdermin=args.gasdermin)
        builder.output_pdb = args.output

        if args.gasdermin:
            print("Gasdermin-specific modifications enabled")

        builder.build_ring(args.n_subunits, args.radius, args.tilt_angle, args.z_offset)

        if args.score:
            score = builder.score_ring()
            print(f"Total score: {score['total_score']:.2f}")
            print(f"Pore quality: {score['pore_quality']:.2f}")
            print(f"H-bonds per subunit: {score['hbond_per_subunit']:.2f}")
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
