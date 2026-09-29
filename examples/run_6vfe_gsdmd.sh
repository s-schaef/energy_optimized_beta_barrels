#!/usr/bin/env bash
# Example 1: human gasdermin D (GSDMD) pore, PDB 6VFE (33 protomers).
#
# Builds a 33-mer ring from a single protomer of the deposited pore:
# extract one chain -> align -> coarse screen -> build with chosen parameters.
# Outputs are written to examples/output/6vfe/.
set -euo pipefail

EXAMPLES="$(cd "$(dirname "$0")" && pwd)"
OUT="$EXAMPLES/output/6vfe"
mkdir -p "$OUT"
cd "$OUT"

# 1. Download 6VFE from the RCSB PDB and extract one protomer (chain A)
python "$EXAMPLES/fetch_protomer.py" 6VFE A gsdmd_protomer.pdb

# 2. Align the protomer: largest beta-sheet along z, centered at the origin
barrel-align --input gsdmd_protomer.pdb --output gsdmd_aligned.pdb

# 3. Coarse screen of radius and tilt angle for a 33-mer
#    (200 scored assemblies; about 2 minutes on 10 cores)
barrel-optimize --monomer gsdmd_aligned.pdb --n_subunits 33 --gasdermin

# 4. Build and score the final model with the chosen parameters
barrel-build --input gsdmd_aligned.pdb --output gsdmd_33mer.pdb \
    --n_subunits 33 --radius 128 --tilt_angle -20 --gasdermin --score
