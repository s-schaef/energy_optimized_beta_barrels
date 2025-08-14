# Protein Ring Assembly Builder

A computational tool for building and optimizing circular protein assemblies from monomeric structures. This package uses PyRosetta scoring functions to optimize ring geometry through parallel grid search, with special support for beta-sheet containing proteins and gasdermin-family proteins.

## Features

- **Automatic monomer alignment**: Aligns protein monomers to a standard orientation using principal component analysis (PCA)
- **Beta-sheet detection**: Identifies and uses beta-sheet structures for improved alignment
- **Parallel optimization**: Uses multiprocessing for efficient parameter space exploration
- **Flexible ring construction**: Supports rings with 2-52 subunits
- **PyRosetta scoring**: Evaluates assemblies using atomic attraction/repulsion and hydrogen bonding terms
- **Gasdermin-specific mode**: Special optimizations for gasdermin-family proteins

## Installation

### Prerequisites

- Miniconda or Anaconda

### Step 1: Create Conda Environment

```bash
# Create a new conda environment with Python 3.9
conda create -n protein-rings python=3.9
conda activate protein-rings
```

### Step 2: Install Dependencies

```bash
# Install scientific computing packages
conda install -c conda-forge numpy pandas scikit-learn

# Install MDAnalysis for structure manipulation
conda install -c conda-forge mdanalysis

# Install Biopython (required by MDAnalysis DSSP module)
conda install -c conda-forge biopython
```

### Step 3: Install PyRosetta

1. Obtain PyRosetta from https://www.pyrosetta.org/downloads (requires free academic license)
2. Download the appropriate wheel file for your system (Python 3.9)
3. Install PyRosetta:

```bash
# Replace with your downloaded PyRosetta wheel file
pip install pyrosetta-2024.*.*.*.wheel
```

### Step 4: Clone the Repository

```bash
git clone https://github.com/yourusername/protein-ring-assembly.git
cd protein-ring-assembly
```

## Usage

The package consists of three main modules that work together:

### 1. Align Your Monomer (`alignment_module.py`)

First, align your monomeric protein structure to a standard orientation:

```bash
python alignment_module.py --input monomer.pdb --output monomer_aligned.pdb
```

This will:
- Detect beta-sheets (if present)
- Align the largest beta-sheet or protein principal axis with the z-axis
- Center the beta-sheet (if not present the center of mass) of the structure at the origin

### 2. Build a Ring (`ring_builder.py`)

Create a circular assembly with specified parameters:

```bash
# Basic ring with 30 subunits
python ring_builder.py --input monomer_aligned.pdb --output ring_30mer.pdb --n_subunits 30

# Ring with custom parameters and scoring
python ring_builder.py --input monomer_aligned.pdb --output ring_custom.pdb \
    --n_subunits 24 --radius 85.0 --tilt_angle 15.0 --score

# For gasdermin proteins
python ring_builder.py --input gasdermin_aligned.pdb --output gasdermin_ring.pdb \
    --n_subunits 30 --gasdermin --score
```

### 3. Optimize Ring Geometry (`optimization_module.py`)

Find optimal ring parameters through parallel grid search:

```bash
# Basic optimization
python optimization_module.py --monomer monomer_aligned.pdb --n_subunits 30

# Custom optimization with specific ranges
python optimization_module.py --monomer monomer_aligned.pdb --n_subunits 24 \
    --radius_range 70 90 --angle_range -20 20 --rounds 3

# Gasdermin optimization with more processes
python optimization_module.py --monomer gasdermin_aligned.pdb --n_subunits 30 \
    --gasdermin --processes 16 --rounds 2
```

## Example Workflow

Here's a complete example for building an optimized 24-mer ring:

```bash
# 1. Align the monomer
python alignment_module.py --input my_protein.pdb --output my_protein_aligned.pdb

# 2. Run optimization to find best parameters
python optimization_module.py --monomer my_protein_aligned.pdb --n_subunits 24 \
    --rounds 2 --processes 8

# 3. Build the final optimized ring (using parameters from optimization output)
python ring_builder.py --input my_protein_aligned.pdb --output final_ring_24mer.pdb \
    --n_subunits 24 --radius 82.5 --tilt_angle 12.3 --score
```

## Output Files

- **Aligned monomer**: `*_aligned.pdb` - Monomer in standard orientation
- **Optimization results**: `round{N}_results.csv` - Scored parameter combinations for each round
- **Final ring**: `optimized_ring_*.pdb` - Best ring assembly found

## Optimization Parameters

The optimization module explores two key parameters:

- **Radius**: Distance from ring center to subunit center (Ångstroms)
- **Tilt angle**: Rotation around x-axis for beta-sheet orientation (degrees)

The scoring function evaluates:
- `fa_atr`: Attractive forces between atoms (weight: 0.9)
- `fa_rep`: Repulsive forces between atoms (weight: 0.02, allowing minor overlaps)
- `hbond_sr_bb`: Short-range backbone hydrogen bonds (weight: 1.0)
- `hbond_lr_bb`: Long-range backbone hydrogen bonds (weight: 10.0, prioritized)

## Command-Line Options

### alignment_module.py
- `--input`: Input PDB file (required)
- `--output`: Output aligned PDB file (optional)

### ring_builder.py
- `--input`: Input aligned monomer PDB (required)
- `--output`: Output ring PDB file (required)
- `--n_subunits`: Number of subunits in ring (default: 30)
- `--radius`: Ring radius in Ångstroms (default: 50.0)
- `--tilt_angle`: Tilt angle in degrees (default: 0.0)
- `--score`: Calculate PyRosetta scores
- `--gasdermin`: Enable gasdermin-specific modifications

### optimization_module.py
- `--monomer`: Aligned monomer PDB file (required)
- `--n_subunits`: Number of subunits in ring (required)
- `--radius_range`: Min and max radius values (default: adaptive)
- `--angle_range`: Min and max tilt angles (default: -30 30)
- `--rounds`: Number of optimization rounds (default: 2)
- `--processes`: Number of parallel processes (default: all cores)
- `--no_csv`: Don't save CSV results
- `--gasdermin`: Enable gasdermin-specific modifications

## Tips for Best Results

1. **Monomer preparation**: Ensure your input monomer is a clean, single-chain structure
2. **Beta-sheet proteins**: The tool works particularly well with beta-barrel and beta-sheet proteins
3. **Optimization rounds**: More rounds give finer results but take longer (2-3 rounds usually sufficient)
4. **Parameter ranges**: Start with default ranges; narrow them based on initial results
5. **Subunit number**: Common ring sizes are 6, 8, 12, 24, 30, 36 subunits

## Troubleshooting

### Common Issues

1. **"No beta strands detected"**: The tool will fall back to geometric alignment. This is normal for alpha-helical proteins.

2. **High repulsion scores**: Try increasing the radius range or adjusting the tilt angle range.

3. **PyRosetta initialization errors**: Ensure PyRosetta is properly licensed and installed for your Python version.

4. **Multiprocessing errors**: Reduce the number of processes with `--processes` or use `--processes 1` for sequential execution.

## Citation

If you use this tool in your research, please cite:
```
[Your publication details here]
```

## License

[Your license here - e.g., MIT, GPL, etc.]

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

## Authors

[Your name and contributors]

## Acknowledgments

This tool uses:
- PyRosetta for energy calculations
- MDAnalysis for structure manipulation
- NumPy, SciPy, and scikit-learn for computational geometry
