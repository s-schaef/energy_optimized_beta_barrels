# Beta-Barrel Assembly Builder

A computational tool for building and optimizing circular beta-barrel protein assemblies from monomeric structures. This package uses modified PyRosetta scoring functions to optimize ring geometry through parallel grid search, with special support for gasdermin-family proteins.

## Features

- **Automatic beta-sheet alignment**: Aligns protein monomers to a standard orientation using principal component analysis (PCA)
- **Beta-sheet detection**: Identifies and uses the largest beta-sheet for improved alignment
- **Pore-quality-focused optimization**: Ranks results by inter-subunit hydrogen bonding (pore quality) rather than raw total energy
- **Z-offset parameter**: Supports alternating vertical stagger between adjacent subunits for proper beta-barrel hydrogen bonding
- **3D parameter search**: Optimizes radius, tilt angle, and z-offset simultaneously
- **Parallel optimization**: Uses multiprocessing for efficient parameter space exploration
- **Flexible ring construction**: Supports rings with 2-52 subunits
- **PyRosetta scoring**: Evaluates assemblies using atomic attraction/repulsion and hydrogen bonding terms
- **Gasdermin-specific mode**: Special optimizations for gasdermin-family proteins

## Installation

### Install conda environment
```bash
git clone https://github.com/s-schaef/energy_optimized_beta_barrels.git
cd energy_optimized_beta_barrels
conda env create -f environment.yml
conda activate energy_optimized_beta_barrels
```

### Install the package

```bash
pip install -e .
```

This registers three CLI commands (`barrel-align`, `barrel-build`, `barrel-optimize`) that work from any directory once the environment is activated.

### Install PyRosetta into your active environment

PyRosetta is free for academic use under the license found here https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.PyRosetta.md

```bash
pip install pyrosetta-installer 
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

## Usage

The package consists of three main modules that work together. After `pip install -e .`, use the `barrel-*` commands from anywhere:

### 1. Align Your Monomer

First, align your monomeric protein structure to a standard orientation:

```bash
barrel-align --input monomer.pdb --output monomer_aligned.pdb
```

This will:
- Detect beta-sheets (if present)
- Align the largest beta-sheet or protein principal axis with the z-axis
- Center the beta-sheet (if not present the center of mass) of the structure at the origin

### 2. (If you already know the geometry) Directly build a Ring

Create a circular assembly with specified parameters:

```bash
# Basic ring with 30 subunits
barrel-build --input monomer_aligned.pdb --output ring_30mer.pdb --n_subunits 30

# Ring with custom parameters including z-offset stagger and scoring
barrel-build --input monomer_aligned.pdb --output ring_custom.pdb \
    --n_subunits 24 --radius 85.0 --tilt_angle -16.0 --z_offset 2.5 --score
```

Empirically, gasdermin assemblies benefit from an additional 10 deg. rotation around the y-axis that results in beta-barrels that are slightly narrower towards the bottom. The --gasdermin flag enables this. 

```bash
# For gasdermin proteins
barrel-build --input monomer_aligned.pdb --output ring_custom.pdb \
    --n_subunits 33 --radius 120.0 --tilt_angle -16.0 --score --gasdermin
```

### 3. (If you don't know the geometry) Search for the best Ring Geometry

Find optimal ring parameters through parallel grid search:

```bash
# Basic optimization (3 rounds, ranked by pore quality)
barrel-optimize --monomer monomer_aligned.pdb --n_subunits 30

# Custom optimization with specific ranges
barrel-optimize --monomer monomer_aligned.pdb --n_subunits 24 \
    --radius_range 70 90 --angle_range -20 20 --z_offset_range 0 4 --rounds 3

# Rank by total energy instead of pore quality (not recommended)
barrel-optimize --monomer monomer_aligned.pdb --n_subunits 24 \
    --rank_by total_score

# Gasdermin optimization
barrel-optimize --monomer gasdermin_aligned.pdb --n_subunits 30 \
    --gasdermin --processes 16
```

## Example Workflow

Here's a complete example for building an optimized 24-mer ring:

```bash
# 1. Align the monomer
barrel-align --input monomer.pdb --output monomer_aligned.pdb

# 2. Run optimization to find best parameters (pore quality focused)
barrel-optimize --monomer monomer_aligned.pdb --n_subunits 24 \
    --rounds 3 --processes 8

# 3. (optional) Build with manually adjusted parameters after visual assessment
barrel-build --input monomer_aligned.pdb --output final_ring_24mer.pdb \
    --n_subunits 24 --radius 82.5 --tilt_angle -12.3 --z_offset 2.0 --score
```

Or use the example workflow script:

```bash
python examples/example_workflow.py --input monomer.pdb --n_subunits 24 --rounds 3
```

## Output Files

- **Aligned monomer**: `aligned_*.pdb` - Monomer in standard orientation
- **Optimization results**: `round{N}_results.csv` - Scored parameter combinations for each round
- **Final ring**: `optimized_ring_*.pdb` - Best ring assembly found

## Optimization Parameters

The optimization module explores three key parameters:

- **Radius**: Distance from ring center to subunit center (Angstroms)
- **Tilt angle**: Beta-sheet rotation around x-axis (degrees). Beta-barrels are often tilted and don't face 'straight down'. 
- **Z-offset**: Alternating vertical displacement between adjacent subunits (Angstroms). In real beta barrels, adjacent strands are staggered along the barrel axis to allow proper hydrogen bonding.

The scoring function evaluates:
- `fa_atr`: Attractive forces between atoms (weight: 0.9)
- `fa_rep`: Repulsive forces between atoms (weight: 0.02, allowing minor overlaps)
- `hbond_sr_bb`: Short-range backbone hydrogen bonds (weight: 1.0)
- `hbond_lr_bb`: Long-range backbone hydrogen bonds (weight: 10.0, prioritized)

The scores are reweighted empirically to recreate some known beta-barrel structures.

### Pore Quality Score

By default, results are ranked by `pore_quality` rather than `total_score`. The pore quality metric heavily weights inter-subunit backbone hydrogen bonds (`hbond_lr_bb`), which are the hallmark of a well-formed beta-barrel pore. This ensures the optimizer finds geometries that produce proper beta-sheet hydrogen bonding across subunits, not just the lowest total energy.

## Command-Line Options

### `barrel-align`
- `--input`: Input PDB file (required)
- `--output`: Output aligned PDB file (optional)

### `barrel-build`
- `--input`: Input aligned monomer PDB (required)
- `--output`: Output ring PDB file (required)
- `--n_subunits`: Number of subunits in ring (default: 30)
- `--radius`: Ring radius in Angstroms (default: 120.0)
- `--tilt_angle`: Tilt angle in degrees (default: -16.0)
- `--z_offset`: Alternating z-offset between adjacent subunits in Angstroms (default: 0.0)
- `--score`: Calculate PyRosetta scores
- `--gasdermin`: Enable 10 degree rotation around the y-axis

### `barrel-optimize`
- `--monomer`: Aligned monomer PDB file (required)
- `--n_subunits`: Number of subunits in ring (required)
- `--radius_range`: Min and max radius values (default: adaptive)
- `--angle_range`: Min and max tilt angles (default: -30 30)
- `--z_offset_range`: Min and max z-offsets (default: 0 5)
- `--grid_size`: Points per dimension per round (default: 8)
- `--rounds`: Number of optimization rounds (default: 3)
- `--rank_by`: Score to rank by: `pore_quality` (default) or `total_score`
- `--processes`: Number of parallel processes (default: all cores)
- `--no_csv`: Don't save CSV results
- `--gasdermin`: Enable gasdermin-specific y-axis rotation

### Python API

After installation, the package can also be used as a library:

```python
from barrel_builder import align_monomer_from_file, RingBuilder, RingOptimizer
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

## Tips for Best Results

1. **Monomer preparation**: Ensure your input monomer is a clean, single-chain structure
2. **Optimization rounds**: More rounds give finer results but take longer (3 rounds usually sufficient)
3. **Parameter ranges**: Start with default ranges; narrow them based on initial results
4. **Z-offset**: For beta-barrels with strong inter-subunit hydrogen bonds, try z_offset values of 1-4 Angstroms
5. **Ranking**: Use `pore_quality` (default) to prioritize pore formation quality over raw energy minimization

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This tool uses:
- PyRosetta for energy calculations
- MDAnalysis for structure manipulation
- NumPy, SciPy, and scikit-learn for computational geometry
