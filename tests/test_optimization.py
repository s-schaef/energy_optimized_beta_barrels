"""Tests for barrel_builder.optimization - parameter grid and screening logic."""

import importlib.util
import pytest
import numpy as np
import pandas as pd

from tests.make_test_pdb import make_simple_monomer_pdb
from barrel_builder.optimization import RingOptimizer
from barrel_builder.ring_builder import RingBuilder, SCORE_WEIGHTS


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

    def test_gasdermin_flag_reaches_builder(self, monomer_pdb):
        opt = RingOptimizer(monomer_pdb, n_subunits=8, gasdermin=True)
        assert opt.gasdermin is True
        assert opt.builder.gasdermin is True

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
            grid_size=5,
        )
        assert len(grid) == 5 ** 2

    def test_grid_covers_range(self, monomer_pdb):
        opt = RingOptimizer(monomer_pdb, n_subunits=8)
        grid = opt._generate_parameter_grid(
            radius_range=(60, 80),
            tilt_angle_range=(-10, 10),
            grid_size=4,
        )
        radii = [p['radius'] for p in grid]
        tilts = [p['tilt_angle'] for p in grid]

        assert min(radii) == pytest.approx(60.0)
        assert max(radii) == pytest.approx(80.0)
        assert min(tilts) == pytest.approx(-10.0)
        assert max(tilts) == pytest.approx(10.0)
        assert set(grid[0]) == {'radius', 'tilt_angle'}

    def test_fixed_parameter_is_not_duplicated(self, monomer_pdb):
        """A range with min == max contributes a single value, not grid_size copies."""
        opt = RingOptimizer(monomer_pdb, n_subunits=8)
        grid = opt._generate_parameter_grid(
            radius_range=(60, 80),
            tilt_angle_range=(-16, -16),
            grid_size=5,
        )
        assert len(grid) == 5


class TestFailureHandling:
    def test_missing_pyrosetta_fails_fast(self, monomer_pdb, monkeypatch):
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, 'find_spec',
            lambda name, *args: None if name == 'pyrosetta' else real_find_spec(name, *args))
        opt = RingOptimizer(monomer_pdb, n_subunits=8)
        with pytest.raises(ImportError, match="PyRosetta"):
            opt.optimize(save_csv=False)

    def test_all_failed_raises(self):
        results = pd.DataFrame([
            {'radius': 50.0, 'tilt_angle': 0.0, 'total_score': np.inf, 'error': 'boom'},
            {'radius': 60.0, 'tilt_angle': 0.0, 'total_score': np.inf, 'error': 'boom'},
        ])
        with pytest.raises(RuntimeError, match="All 2 evaluations failed"):
            RingOptimizer._check_failures(results)

    def test_partial_failure_warns(self, capsys):
        results = pd.DataFrame([
            {'radius': 50.0, 'tilt_angle': 0.0, 'total_score': -10.0, 'error': np.nan},
            {'radius': 60.0, 'tilt_angle': 0.0, 'total_score': np.inf, 'error': 'boom'},
        ])
        RingOptimizer._check_failures(results)
        assert "1 of 2 evaluations failed" in capsys.readouterr().out


class TestScoring:
    def test_score_components_sum_to_total(self, monomer_pdb):
        """score_ring() returns weighted components that add up to total_score."""
        pytest.importorskip("pyrosetta")
        builder = RingBuilder(monomer_pdb)
        builder.build_ring(n_subunits=4, radius=50.0)
        scores = builder.score_ring()
        assert set(SCORE_WEIGHTS) <= set(scores)
        assert scores['n_subunits'] == 4
        assert sum(scores[t] for t in SCORE_WEIGHTS) == pytest.approx(scores['total_score'])
