"""Tests for ring_builder.py - focus on ring geometry and pore quality."""

import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.make_test_pdb import make_simple_monomer_pdb, make_helix_monomer_pdb
from ring_builder import RingBuilder


@pytest.fixture
def monomer_pdb(tmp_path):
    """Create a simple test monomer PDB."""
    pdb = str(tmp_path / "monomer.pdb")
    make_simple_monomer_pdb(pdb, n_strands=4, strand_length=8)
    return pdb


@pytest.fixture
def helix_pdb(tmp_path):
    """Create a helix-only test monomer PDB."""
    pdb = str(tmp_path / "helix.pdb")
    make_helix_monomer_pdb(pdb, n_residues=30)
    return pdb


class TestRingBuilderInit:
    def test_init_loads_monomer(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        assert len(builder.monomer_atoms) > 0

    def test_init_missing_file(self):
        with pytest.raises(FileNotFoundError):
            RingBuilder("/nonexistent/file.pdb")

    def test_gasdermin_flag_stored(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb, gasdermin=True)
        assert builder.gasdermin is True

    def test_gasdermin_default_false(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        assert builder.gasdermin is False


class TestBuildRing:
    def test_basic_ring(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        ring = builder.build_ring(n_subunits=4, radius=50.0)
        assert ring is not None
        assert len(ring.segments) == 4

    def test_subunit_count(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        for n in [2, 5, 10, 26]:
            ring = builder.build_ring(n_subunits=n, radius=50.0)
            assert len(ring.segments) == n

    def test_invalid_subunits(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        with pytest.raises(ValueError, match="at least 2"):
            builder.build_ring(n_subunits=1)
        with pytest.raises(ValueError, match="cannot exceed 52"):
            builder.build_ring(n_subunits=53)

    def test_invalid_radius(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        with pytest.raises(ValueError, match="positive"):
            builder.build_ring(n_subunits=4, radius=-10)

    def test_ring_is_circular(self, monomer_pdb):
        """Subunit centers should lie approximately on a circle of the given radius."""
        builder = RingBuilder(monomer_pdb)
        radius = 60.0
        ring = builder.build_ring(n_subunits=8, radius=radius)

        for seg in ring.segments:
            atoms = ring.select_atoms(f"segid {seg.segid}")
            com = atoms.center_of_mass()
            # Distance from ring center (approximately origin) in xy plane
            r_xy = np.sqrt(com[0] ** 2 + com[1] ** 2)
            assert abs(r_xy - radius) < 15.0, (
                f"Segment {seg.segid} COM r_xy={r_xy:.1f} far from radius={radius}"
            )

    def test_z_offset_alternates(self, monomer_pdb):
        """With z_offset > 0, adjacent subunits should be at different z heights."""
        builder = RingBuilder(monomer_pdb)
        z_off = 3.0
        ring = builder.build_ring(n_subunits=6, radius=50.0, z_offset=z_off)

        z_centers = []
        for seg in ring.segments:
            atoms = ring.select_atoms(f"segid {seg.segid}")
            z_centers.append(atoms.center_of_mass()[2])

        # Adjacent subunits should differ in z
        for i in range(len(z_centers) - 1):
            diff = abs(z_centers[i] - z_centers[i + 1])
            assert diff > 0.5, (
                f"Adjacent z-centers too close: {z_centers[i]:.2f} vs {z_centers[i+1]:.2f}"
            )

    def test_z_offset_zero_is_flat(self, monomer_pdb):
        """With z_offset=0, all subunits should be in roughly the same z plane."""
        builder = RingBuilder(monomer_pdb)
        ring = builder.build_ring(n_subunits=6, radius=50.0, z_offset=0.0)

        z_centers = []
        for seg in ring.segments:
            atoms = ring.select_atoms(f"segid {seg.segid}")
            z_centers.append(atoms.center_of_mass()[2])

        z_range = max(z_centers) - min(z_centers)
        # Should be near zero (only structural extent, not stagger)
        assert z_range < 1.0, f"z_offset=0 but z range is {z_range:.2f}"

    def test_unique_segment_ids(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        ring = builder.build_ring(n_subunits=10, radius=50.0)
        segids = [seg.segid for seg in ring.segments]
        assert len(segids) == len(set(segids))

    def test_tilt_changes_orientation(self, monomer_pdb):
        """Different tilt angles should produce different structures."""
        builder = RingBuilder(monomer_pdb)

        ring1 = builder.build_ring(n_subunits=4, radius=50.0, tilt_angle=0.0)
        pos1 = ring1.atoms.positions.copy()

        ring2 = builder.build_ring(n_subunits=4, radius=50.0, tilt_angle=20.0)
        pos2 = ring2.atoms.positions.copy()

        assert not np.allclose(pos1, pos2, atol=0.1)


class TestWriteAndCenter:
    def test_write_ring(self, monomer_pdb, tmp_path):
        builder = RingBuilder(monomer_pdb)
        builder.build_ring(n_subunits=4, radius=50.0)
        out = str(tmp_path / "ring.pdb")
        builder.write_ring_pdb(out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_center_ring(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        builder.build_ring(n_subunits=4, radius=50.0)
        builder.center_ring()
        com = builder.ring.atoms.center_of_mass()
        assert np.allclose(com, [0, 0, 0], atol=0.1)

    def test_write_before_build_raises(self, monomer_pdb, tmp_path):
        builder = RingBuilder(monomer_pdb)
        with pytest.raises(RuntimeError):
            builder.write_ring_pdb(str(tmp_path / "fail.pdb"))

    def test_center_before_build_raises(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        with pytest.raises(RuntimeError):
            builder.center_ring()


class TestRingInfo:
    def test_ring_info(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        builder.build_ring(n_subunits=4, radius=50.0)
        info = builder.get_ring_info()
        assert info is not None
        assert info['n_segments'] == 4
        assert info['n_atoms'] > 0

    def test_ring_info_before_build(self, monomer_pdb):
        builder = RingBuilder(monomer_pdb)
        assert builder.get_ring_info() is None
