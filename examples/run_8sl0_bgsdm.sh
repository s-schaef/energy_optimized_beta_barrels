#!/usr/bin/env bash
# Example 2: Vitiosangium bacterial gasdermin (bGSDM), PDB 8SL0 (52-mer pore).
#
# Builds a 52-mer ring from the protomer of the 'slinky'-like oligomer
# structure: extract the chain -> align -> coarse screen -> build with
# chosen parameters. Outputs are written to examples/output/8sl0/.
set -euo pipefail

EXAMPLES="$(cd "$(dirname "$0")" && pwd)"
OUT="$EXAMPLES/output/8sl0"
mkdir -p "$OUT"
cd "$OUT"

# 1. Download 8SL0 from the RCSB PDB (mmCIF only) and extract the protomer
python "$EXAMPLES/fetch_protomer.py" 8SL0 A bgsdm_protomer.pdb

# 2. Align the protomer: largest beta-sheet along z, centered at the origin
barrel-align --input bgsdm_protomer.pdb --output bgsdm_aligned.pdb

# 3. Coarse screen of radius and tilt angle for a 52-mer
#    (200 scored assemblies; about 2-3 minutes on 10 cores)
barrel-optimize --monomer bgsdm_aligned.pdb --n_subunits 52

# 4. Build and score the final model with the chosen parameters
barrel-build --input bgsdm_aligned.pdb --output bgsdm_52mer.pdb \
    --n_subunits 52 --radius 179 --tilt_angle -17 --score
