"""Tests for barrel_builder.alignment."""

import os
import pytest
import numpy as np
import MDAnalysis as mda
from scipy.spatial.transform import Rotation

from tests.make_test_pdb import make_simple_monomer_pdb, make_helix_monomer_pdb
from barrel_builder.alignment import MonomerAligner, align_monomer_from_file

# Chain A of PDB 6VFE (human GSDMD pore; Xia et al., Nature 2021)
GSDMD_PDB = os.path.join(os.path.dirname(__file__), 'data', '6vfe_chainA.pdb')


@pytest.fixture
def beta_pdb(tmp_path):
    pdb = str(tmp_path / "beta_monomer.pdb")
    make_simple_monomer_pdb(pdb, n_strands=4, strand_length=8, spacing=5.0)
    return pdb


@pytest.fixture
def helix_pdb(tmp_path):
    pdb = str(tmp_path / "helix_monomer.pdb")
    make_helix_monomer_pdb(pdb, n_residues=30)
    return pdb


def rotated_copy(pdb, out, seed):
    """Write a randomly rotated and translated copy of a PDB file."""
    u = mda.Universe(pdb)
    rot = Rotation.random(random_state=seed).as_matrix()
    u.atoms.positions = u.atoms.positions @ rot.T + np.array([10.0, -20.0, 30.0])
    u.atoms.write(out)
    return out


def ca_chirality(atoms):
    """Sign of the N-CA-C-CB signed volume for every residue with a CB (+1 for L)."""
    signs = []
    for res in atoms.residues:
        names = {a.name: a.position for a in res.atoms}
        if all(n in names for n in ('N', 'CA', 'C', 'CB')):
            ca = names['CA']
            signs.append(np.sign(np.dot(names['N'] - ca,
                                        np.cross(names['C'] - ca, names['CB'] - ca))))
    return np.array(signs)


class TestMonomerAligner:
    def test_loads_protein(self, beta_pdb):
        aligner = MonomerAligner(beta_pdb)
        assert len(aligner.protein) > 0

    def test_secondary_structure_detected(self, beta_pdb):
        aligner = MonomerAligner(beta_pdb)
        assert len(aligner.ss_dict) == len(aligner.protein.residues)

    def test_helix_fallback(self, helix_pdb, tmp_path):
        """Without beta strands, alignment falls back to whole-protein PCA."""
        aligner = MonomerAligner(helix_pdb)
        assert not aligner.has_beta_sheet
        aligned = aligner.align_to_standard_orientation(str(tmp_path / "aligned.pdb"))
        assert np.allclose(aligned.center_of_mass(), 0, atol=1e-3)

    def test_real_protomer_uses_largest_sheet(self):
        aligner = MonomerAligner(GSDMD_PDB)
        assert aligner.has_beta_sheet
        assert aligner.largest_sheet_info['n_strands'] >= 4

    def test_multichain_input_raises(self, tmp_path):
        u = mda.Universe(GSDMD_PDB)
        second = u.copy()
        second.atoms.chainIDs = 'B'
        second.atoms.translate([50.0, 0.0, 0.0])
        two_chains = str(tmp_path / "two_chains.pdb")
        mda.Merge(u.atoms, second.atoms).atoms.write(two_chains)
        with pytest.raises(ValueError, match="protein chains"):
            MonomerAligner(two_chains)

    def test_align_writes_output(self, beta_pdb, tmp_path):
        out = str(tmp_path / "aligned.pdb")
        aligner = MonomerAligner(beta_pdb)
        aligner.align_to_standard_orientation(out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_align_default_output(self, beta_pdb, tmp_path, monkeypatch):
        """Without output_file, writes aligned_<input basename> to the working directory."""
        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        MonomerAligner(beta_pdb).align_to_standard_orientation()
        assert (workdir / f"aligned_{os.path.basename(beta_pdb)}").exists()

    def test_convenience_function(self, beta_pdb, tmp_path):
        out = str(tmp_path / "aligned2.pdb")
        aligner = align_monomer_from_file(beta_pdb, out)
        assert os.path.exists(out)
        assert isinstance(aligner, MonomerAligner)

    def test_get_aligned_monomer(self, beta_pdb):
        aligner = MonomerAligner(beta_pdb)
        atoms = aligner.get_aligned_monomer()
        assert len(atoms) > 0


class TestAlignmentGeometry:
    def test_standard_orientation(self, tmp_path):
        """Sheet centered at the origin, protein body at +x and +z."""
        aligner = MonomerAligner(GSDMD_PDB)
        aligned = aligner.align_to_standard_orientation(str(tmp_path / "aligned.pdb"))
        resids = aligner.largest_sheet_info['sheet_resids']
        sheet = aligned.select_atoms(f'resid {" ".join(map(str, resids))} and backbone')
        assert np.allclose(sheet.center_of_mass(), 0, atol=1e-3)
        assert aligned.center_of_mass()[0] > 0
        assert aligned.center_of_mass()[2] > 0

    @pytest.mark.parametrize("seed", [0, 4, 5])
    def test_alignment_never_mirrors(self, tmp_path, seed):
        """
        Regression test: for any input orientation the aligned protomer keeps
        its chirality and ends up in the same standard orientation.
        (Seeds 0, 4 and 5 produced mirrored output before the fix.)
        """
        reference = MonomerAligner(GSDMD_PDB).align_to_standard_orientation(
            str(tmp_path / "aligned_ref.pdb")).positions.copy()

        rotated = rotated_copy(GSDMD_PDB, str(tmp_path / f"rot{seed}.pdb"), seed)
        aligned = MonomerAligner(rotated).align_to_standard_orientation(
            str(tmp_path / f"aligned_rot{seed}.pdb"))

        assert (ca_chirality(aligned) > 0).all()
        assert np.abs(aligned.positions - reference).max() < 0.1
