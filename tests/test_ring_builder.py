"""Tests for ring_builder.py - focus on ring geometry and pore quality."""

import os
import pytest
import numpy as np

from tests.make_test_pdb import make_simple_monomer_pdb, make_helix_monomer_pdb
from barrel_builder.ring_builder import RingBuilder


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
        """Subunit centers of geometry lie on a circle of the given radius."""
        builder = RingBuilder(monomer_pdb)
        radius = 60.0
        ring = builder.build_ring(n_subunits=8, radius=radius)

        centers = np.array([
            ring.select_atoms(f"segid {seg.segid}").center_of_geometry()
            for seg in ring.segments
        ])
        ring_center = centers.mean(axis=0)
        r_xy = np.linalg.norm((centers - ring_center)[:, :2], axis=1)
        assert np.allclose(r_xy, radius, atol=1e-3)
        assert np.allclose(centers[:, 2], centers[0, 2], atol=1e-3)

    def test_ring_is_symmetric(self, monomer_pdb):
        """Rotating subunit i by 360/n degrees around the ring axis gives subunit i+1."""
        builder = RingBuilder(monomer_pdb)
        n = 6
        ring = builder.build_ring(n_subunits=n, radius=50.0, tilt_angle=-15.0)
        builder.center_ring()

        angle = np.radians(360 / n)
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])
        segids = [seg.segid for seg in ring.segments]
        first = ring.select_atoms(f"segid {segids[0]}").positions
        second = ring.select_atoms(f"segid {segids[1]}").positions
        assert np.allclose(first @ rot.T, second, atol=1e-3)

    def test_cone_angle_default_is_zero(self, helix_pdb):
        builder = RingBuilder(helix_pdb)
        default = builder.build_ring(n_subunits=4, radius=50.0).atoms.positions.copy()
        zero = builder.build_ring(n_subunits=4, radius=50.0, cone_angle=0.0).atoms.positions
        assert np.allclose(default, zero)

    def test_cone_angle_changes_geometry(self, helix_pdb):
        builder = RingBuilder(helix_pdb)
        plain = builder.build_ring(n_subunits=4, radius=50.0).atoms.positions.copy()
        coned = builder.build_ring(n_subunits=4, radius=50.0, cone_angle=10.0).atoms.positions
        assert not np.allclose(plain, coned, atol=0.1)

    def test_positive_cone_angle_narrows_bottom(self, helix_pdb):
        """With a positive cone angle, the bottom (-z) of each subunit moves towards the axis."""
        builder = RingBuilder(helix_pdb)

        def radial_distance(cone_angle):
            builder.build_ring(n_subunits=6, radius=50.0, cone_angle=cone_angle)
            builder.center_ring()
            positions = builder.ring.select_atoms("segid A").positions
            return np.linalg.norm(positions[:, :2], axis=1), positions[:, 2]

        r_flat, z_flat = radial_distance(0.0)
        r_cone, _ = radial_distance(10.0)
        bottom = z_flat < np.median(z_flat)
        assert r_cone[bottom].mean() < r_flat[bottom].mean()
        assert r_cone[~bottom].mean() > r_flat[~bottom].mean()

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
