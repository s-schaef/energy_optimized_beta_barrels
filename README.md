# Beta-Barrel Assembly Builder

`barrel-builder` builds circular (C<sub>n</sub>-symmetric) models of β-barrel pores, such as gasdermin pores, from the structure of a single protomer. It aligns the protomer, arranges n rigid copies on a ring and screens ring geometries (radius and tilt angle) with an empirically reweighted PyRosetta score.

## Scope and limitations

This is a tool for **coarse structure identification**:

- The models are **rigid-body assemblies of the unmodified protomer**. They are **not energy-minimized** or otherwise relaxed, and the protomer conformation is never changed.
- The PyRosetta energy terms are **empirically reweighted** (`fa_atr` ×0.9, `fa_rep` ×0.02, `hbond_sr_bb` ×1.0, `hbond_lr_bb` ×10.0). This tolerates small overlaps between the rigid protomers and favours backbone hydrogen bonds. The resulting scores are not physical energies and only compare geometries built from the same protomer.
- Use the score as a **coarse screen during geometry exploration**, not to select the final parameters.
- **Evaluate every resulting structure manually** against known insights (stoichiometry, homologous pore structures, membrane insertion) and against experimental data such as a cryo-EM density.

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/s-schaef/energy_optimized_beta_barrels.git
cd energy_optimized_beta_barrels
conda env create -f environment.yml
conda activate energy_optimized_beta_barrels
pip install -e .
```

This installs the command-line tools `barrel-align`, `barrel-build` and `barrel-optimize`.

Scoring (`barrel-optimize` and `barrel-build --score`) requires [PyRosetta](https://www.pyrosetta.org), which is free for academic use under its [license](https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.PyRosetta.md). The environment already contains the installer:

```bash
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

Without conda, `pip install -e .` in any Python ≥ 3.10 environment works as well; PyRosetta is then installed with `pip install pyrosetta-installer` followed by the command above.

## Quick start

```bash
# 1. Align a single protomer (one chain; see examples/fetch_protomer.py)
barrel-align --input protomer.pdb --output protomer_aligned.pdb

# 2. Screen radius and tilt angle for a ring of 33 protomers
barrel-optimize --monomer protomer_aligned.pdb --n_subunits 33 --cone_angle 10

# 3. Build (and score) the ring with the parameters you settle on
barrel-build --input protomer_aligned.pdb --output ring_33mer.pdb \
    --n_subunits 33 --radius 128 --tilt_angle -20 --cone_angle 10 --score
```

## Examples

[examples/](examples/README.md) contains two complete workflows based on published structures:

- `run_6vfe_gsdmd.sh`: the human GSDMD pore (PDB 6VFE, 33-mer)
- `run_8sl0_bgsdm.sh`: the *Vitiosangium* bacterial gasdermin pore (PDB 8SL0, 52-mer)

## How it works

### 1. Alignment (`barrel-align`)

DSSP (via MDAnalysis) assigns β-strands. Strands linked by backbone hydrogen bonds are grouped into sheets, and the largest sheet is used for the alignment:

- its first principal axis is aligned with z and its second with y;
- its center of mass is placed at the origin;
- the protein is flipped so that the rest of the protein lies at +x and +z.

If no β-strands are found, the Cα atoms of the whole protein are used instead. The input must be a single protein chain.

### 2. Ring geometry (`barrel-build`)

Each copy of the aligned protomer is rotated about its own center of geometry:

1. by the **tilt angle** around the x-axis (the radial direction), which inclines the β-strands relative to the pore axis;
2. by the **cone angle** around the y-axis (the tangential direction). Positive values make the ring narrower at the bottom (−z; after alignment the bulk of the protomer lies at +z). The default is 0°;
3. by 360°·i/n around the z-axis, to face the ring axis.

A suitable cone angle depends on the pore. We compared 0° and 10° by the Cα RMSD to the published pores, each at its best-fitting radius and tilt:
- the GSDMD pore 6VFE is reproduced much better at 10° (0.6 Å, against 4.7 Å at 0°);
- the *Vitiosangium* bGSDM pore models 9A84 and 9A85 are reproduced much better at 0° (1.5 Å, against 5.6 Å at 10°).

Choose the cone angle based on experimental data.

The copy is then placed at a distance **radius** from the ring axis. The radius is therefore the distance from the ring axis to each protomer's center of geometry. It is not the radius of the pore lumen, which is smaller. The assembled ring is centered at the origin with the pore axis along z. Up to 52 protomers are supported (chain IDs A–Z, a–z).

### 3. Scoring and screening (`barrel-optimize`)

Each assembly is scored as is, with no minimization or repacking, using the reweighted PyRosetta terms listed above. `total_score` is their weighted sum; lower is better.

The screen evaluates a grid of radius × tilt angle, 10 × 10 by default, in parallel. Each further round is centered on the mean of the two best geometries, with the range narrowed to ± one grid step. The default radius range is 60–100 % of an estimate in which the protomers' widths along y just add up to the ring circumference. The best evaluated geometry is written as a PDB file and should be treated as a starting point for manual evaluation.

## Command-line reference

### `barrel-align`
- `--input`: single-protomer PDB file (required)
- `--output`: aligned PDB file (default: `aligned_<input name>` in the working directory)

### `barrel-build`
- `--input`: aligned protomer PDB file (required)
- `--output`: output PDB file for the ring (required)
- `--n_subunits`: number of protomers, 2–52 (default: 30)
- `--radius`: distance from ring axis to protomer center of geometry in Å (default: 120.0)
- `--tilt_angle`: tilt angle around the x-axis in degrees (default: -16.0)
- `--cone_angle`: rotation around the y-axis in degrees; positive values narrow the bottom of the ring (default: 0.0)
- `--score`: score the ring with PyRosetta and print the weighted terms

### `barrel-optimize`
- `--monomer`: aligned protomer PDB file (required)
- `--n_subunits`: number of protomers, 2–52 (required)
- `--radius_range`: min and max radius in Å (default: 60–100 % of the estimated radius)
- `--angle_range`: min and max tilt angle in degrees (default: -30 30)
- `--grid_size`: grid points per dimension and round (default: 10)
- `--rounds`: number of refinement rounds (default: 2)
- `--processes`: number of parallel processes (default: all cores)
- `--cone_angle`: cone angle in degrees, kept fixed during the screen (default: 0.0)
- `--no_csv`: do not write the per-round CSV files

The scripts `alignment_module.py`, `ring_builder.py` and `optimization_module.py` in the repository root accept the same options as the corresponding commands.

## Output files

- `barrel-align`: the aligned protomer
- `barrel-optimize`, written to the working directory:
  - `round{N}_results.csv` (N = 0, 1, ...): every evaluated geometry with `radius`, `tilt_angle`, `total_score` and the weighted terms `fa_atr`, `fa_rep`, `hbond_sr_bb` and `hbond_lr_bb`, sorted by `total_score`
  - `optimized_ring_<radius>A_<tilt>deg.pdb`: the best evaluated geometry
- `barrel-build`: the ring; with `--score`, the weighted terms are printed

## Python API

```python
from barrel_builder import align_monomer_from_file, RingBuilder, RingOptimizer

align_monomer_from_file("protomer.pdb", "protomer_aligned.pdb")

builder = RingBuilder("protomer_aligned.pdb")
builder.build_ring(n_subunits=33, radius=128.0, tilt_angle=-20.0, cone_angle=10.0)
builder.write_ring_pdb("ring_33mer.pdb")
scores = builder.score_ring()  # requires PyRosetta
```

See [examples/example_workflow.py](examples/example_workflow.py) for the complete workflow.

## Testing

```bash
python -m pytest tests/
```

The scoring test is skipped if PyRosetta is not installed.

## References for the example structures

- **6VFE**: Xia, S., Zhang, Z., Magupalli, V.G., Pablo, J.L., Dong, Y., Vora, S.M., Wang, L., Fu, T.M., Jacobson, M.P., Greka, A., Lieberman, J., Ruan, J. & Wu, H. Gasdermin D pore structure reveals preferential release of mature interleukin-1. *Nature* **593**, 607–611 (2021). https://doi.org/10.1038/s41586-021-03478-3
- **8SL0** and the integrative 52-mer pore models **9A84** and **9A85** (equilibrated; PDB-IHM): Johnson, A.G., Mayer, M.L., Schaefer, S.L., McNamara-Bordewick, N.K., Hummer, G. & Kranzusch, P.J. Structure and assembly of a bacterial gasdermin pore. *Nature* **628**, 657–663 (2024). https://doi.org/10.1038/s41586-024-07216-3

## License

This project is licensed under the BSD 3-Clause License; see the [LICENSE](LICENSE) file for details. PyRosetta is subject to its own license.

## Acknowledgments

This tool uses PyRosetta for scoring, MDAnalysis (including its DSSP implementation) for structure handling, Biopython for reading PDB/mmCIF files, and NumPy, SciPy, scikit-learn and pandas.
