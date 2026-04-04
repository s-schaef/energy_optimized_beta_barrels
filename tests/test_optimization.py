"""Tests for optimization_module.py - geometry and parameter grid logic."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.make_test_pdb import make_simple_monomer_pdb
from optimization_module import RingOptimizer


@pytest.fixture
def monomer_pdb(tmp_path):
    pdb = str(tmp_path / "monomer.pdb")
    make_simple_monomer_pdb(pdb, n_strands=4, strand_length=8)
    return pdb


class TestRingOptimizer:
    def test_init(self, monomer_pdb):
        opt = RingOptimizer(monomer_pdb, n_subunits=8)
        assert opt.base_radius > 0
        assert opt.n_subunits == 8

    def test_base_radius_scales_with_subunits(self, monomer_pdb):
        opt8 = RingOptimizer(monomer_pdb, n_subunits=8)
        opt16 = RingOptimizer(monomer_pdb, n_subunits=16)
        assert opt16.base_radius > opt8.base_radius

    def test_gasdermin_flag(self, monomer_pdb):
        opt = RingOptimizer(monomer_pdb, n_subunits=8, gasdermin=True)
        assert opt.gasdermin is True

    def test_n_processes_capped(self, monomer_pdb):
        import multiprocessing as mp
        opt = RingOptimizer(monomer_pdb, n_subunits=8, n_processes=9999)
        assert opt.n_processes <= mp.cpu_count()


class TestParameterGrid:
    def test_grid_size(self, monomer_pdb):
        opt = RingOptimizer(monomer_pdb, n_subunits=8)
        grid = opt._generate_parameter_grid(
            radius_range=(50, 100),
            tilt_angle_range=(-20, 20),
            z_offset_range=(0, 5),
            grid_size=5,
        )
        assert len(grid) == 5 ** 3  # 125 combinations

    def test_grid_covers_range(self, monomer_pdb):
        opt = RingOptimizer(monomer_pdb, n_subunits=8)
        grid = opt._generate_parameter_grid(
            radius_range=(60, 80),
            tilt_angle_range=(-10, 10),
            z_offset_range=(0, 3),
            grid_size=4,
        )
        radii = [p['radius'] for p in grid]
        tilts = [p['tilt_angle'] for p in grid]
        z_offs = [p['z_offset'] for p in grid]

        assert min(radii) == pytest.approx(60.0)
        assert max(radii) == pytest.approx(80.0)
        assert min(tilts) == pytest.approx(-10.0)
        assert max(tilts) == pytest.approx(10.0)
        assert min(z_offs) == pytest.approx(0.0)
        assert max(z_offs) == pytest.approx(3.0)

    def test_grid_params_have_all_keys(self, monomer_pdb):
        opt = RingOptimizer(monomer_pdb, n_subunits=8)
        grid = opt._generate_parameter_grid(
            radius_range=(50, 100),
            tilt_angle_range=(-20, 20),
            z_offset_range=(0, 5),
            grid_size=3,
        )
        for params in grid:
            assert 'radius' in params
            assert 'tilt_angle' in params
            assert 'z_offset' in params


class TestPoreQualityRanking:
    """Verify that the pore_quality score prioritizes H-bonds over raw energy."""

    def test_pore_quality_in_score_dict(self, monomer_pdb):
        """RingBuilder.score_ring() must return pore_quality and hbond_per_subunit."""
        pytest.importorskip("pyrosetta")
        from ring_builder import RingBuilder
        builder = RingBuilder(monomer_pdb)
        builder.build_ring(n_subunits=4, radius=50.0)
        scores = builder.score_ring()
        assert 'pore_quality' in scores
        assert 'hbond_per_subunit' in scores
        assert 'hbond_lr_bb' in scores
