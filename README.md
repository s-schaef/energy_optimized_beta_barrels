# Beta-Barrel Assembly Builder

A computational tool for building and optimizing circular beta-barrel protein assemblies from monomeric structures. This package uses modified PyRosetta scoring functions to optimize ring geometry through parallel grid search, with special support for gasdermin-family proteins.

## Features

- **Automatic beta-sheet alignment**: Aligns protein monomers to a standard orientation using principal component analysis (PCA)
- **Beta-sheet detection**: Identifies and uses the largest beta-sheet for improved alignment
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



### Install PyRosetta into your active environment

PyRosetta is free for academic use under the license found here https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.PyRosetta.md


```bash
pip install pyrosetta-installer 
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

## Usage

The package consists of three main modules that work together:

### 1. Align Your Monomer

First, align your monomeric protein structure to a standard orientation:

```bash
python alignment_module.py --input monomer.pdb --output monomer_aligned.pdb
```

This will:
- Detect beta-sheets (if present)
- Align the largest beta-sheet or protein principal axis with the z-axis
- Center the beta-sheet (if not present the center of mass) of the structure at the origin

### 2. (If you already know the geometry) Directly build a Ring

Create a circular assembly with specified parameters:

```bash
# Basic ring with 30 subunits and a default radius of 120A and default tilt angle of -16 degrees
python ring_builder.py --input monomer_aligned.pdb --output ring_30mer.pdb --n_subunits 30

# Ring with custom parameters and scoring
python ring_builder.py --input monomer_aligned.pdb --output ring_custom.pdb \
    --n_subunits 24 --radius 85.0 --tilt_angle -16.0 --score
```

Empirically, gasdermin assemblies benefit from an additional 10 deg. rotation around the y-axis that results in beta-barrels that are slightly narrower towards the bottom. The --gasdermin flag enables this. 

```bash
# For gasdermin proteins
python ring_builder.py --input monomer_aligned.pdb --output ring_custom.pdb \
    --n_subunits 33 --radius 120.0 --tilt_angle -16.0 --score --gasdermin
```

### 3. (If you don't know the geometry) Search for the best Ring Geometry

Find optimal ring parameters through parallel grid search:

```bash
# Basic optimization (with 2 default optimization rounds)
python optimization_module.py --monomer monomer_aligned.pdb --n_subunits 30

# Custom optimization with specific ranges and one extra optimization iteration
python optimization_module.py --monomer monomer_aligned.pdb --n_subunits 24 \
    --radius_range 70 90 --angle_range -20 20 --rounds 3

# Gasdermin optimization
python optimization_module.py --monomer gasdermin_aligned.pdb --n_subunits 30 \
    --gasdermin --processes 16
```

## Example Workflow

Here's a complete example for building an optimized 24-mer ring:

```bash
# 1. Align the monomer
python alignment_module.py --input monomer.pdb --output monomer_aligned.pdb

# 2. Run optimization to find best parameters 
python optimization_module.py --monomer monomer_aligned.pdb --n_subunits 24 \
    --rounds 2 --processes 8

# 3. (optional and ideally not needed) 
# After visual assessment of the structure, you may want to play around with user specified radius and tilt_angle values. 
python ring_builder.py --input monomer_aligned.pdb --output final_ring_24mer.pdb \
    --n_subunits 24 --radius 82.5 --tilt_angle -12.3 --score
```

## Output Files

- **Aligned monomer**: `aligned_*.pdb` - Monomer in standard orientation
- **Optimization results**: `round{N}_results.csv` - Scored parameter combinations for each round
- **Final ring**: `optimized_ring_*.pdb` - Best ring assembly found

## Optimization Parameters

The optimization module explores two key parameters:

- **Radius**: Distance from ring center to subunit center (Ångstroms)
- **Tilt angle**: Beta-sheet rotation around x-axis (degrees). Beta-barrels are often tilted and don't face 'straight down'. 

The scoring function evaluates:
- `fa_atr`: Attractive forces between atoms (weight: 0.9)
- `fa_rep`: Repulsive forces between atoms (weight: 0.02, allowing minor overlaps)
- `hbond_sr_bb`: Short-range backbone hydrogen bonds (weight: 1.0)
- `hbond_lr_bb`: Long-range backbone hydrogen bonds (weight: 10.0, prioritized)
The scores are reweighted empirically to recreate some known beta-barrel structures. 

## Command-Line Options

### alignment_module.py
- `--input`: Input PDB file (required)
- `--output`: Output aligned PDB file (optional)

### ring_builder.py
- `--input`: Input aligned monomer PDB (required)
- `--output`: Output ring PDB file (required)
- `--n_subunits`: Number of subunits in ring (default: 30)
- `--radius`: Ring radius in Ångstroms (default: 120.0)
- `--tilt_angle`: Tilt angle in degrees (default: -16.0)
- `--score`: Calculate PyRosetta scores
- `--gasdermin`: Enable 10 degree rotation around the y-axis. This leads to beta-barrels that get narrower towards the bottom, as can be seen in some known gasdermin family proteins. 

### optimization_module.py
- `--monomer`: Aligned monomer PDB file (required)
- `--n_subunits`: Number of subunits in ring (required)
- `--radius_range`: Min and max radius values (default: adaptive)
- `--angle_range`: Min and max tilt angles (default: -30 30)
- `--rounds`: Number of optimization rounds (default: 2)
- `--processes`: Number of parallel processes (default: all cores)
- `--no_csv`: Don't save CSV results
- `--gasdermin`: Enable 10 degree rotation around the y-axis. This leads to beta-barrels that get narrower towards the bottom, as can be seen in some known gasdermin family proteins. 

## Tips for Best Results

1. **Monomer preparation**: Ensure your input monomer is a clean, single-chain structure
2. **Optimization rounds**: More rounds give finer results but take longer (2-3 rounds usually sufficient)
3. **Parameter ranges**: Start with default ranges; narrow them based on initial results


## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

This tool uses:
- PyRosetta for energy calculations
- MDAnalysis for structure manipulation
- NumPy, SciPy, and scikit-learn for computational geometry
