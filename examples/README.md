# Examples

Two complete workflows on published gasdermin pore structures. Each script downloads the entry from the RCSB PDB, extracts one protomer, aligns it, runs the coarse screen and builds the final model. The outputs go to `examples/output/<entry>/`, which is not tracked by git.

```bash
conda activate energy_optimized_beta_barrels
bash examples/run_6vfe_gsdmd.sh
bash examples/run_8sl0_bgsdm.sh
```

The scripts need an internet connection and PyRosetta. The screens take a few minutes; set `--processes` in the scripts to limit the number of cores.

## Example 1: human GSDMD pore (PDB 6VFE, 33-mer)

[`run_6vfe_gsdmd.sh`](run_6vfe_gsdmd.sh)

| Step | Command | Output |
|---|---|---|
| 1 | `fetch_protomer.py 6VFE A` | `6VFE.pdb`, `gsdmd_protomer.pdb` (residues 3–241) |
| 2 | `barrel-align` | `gsdmd_aligned.pdb` |
| 3 | `barrel-optimize --n_subunits 33 --gasdermin` | `round0_results.csv`, `round1_results.csv`, `optimized_ring_*.pdb` |
| 4 | `barrel-build --radius 128 --tilt_angle -20 --gasdermin --score` | `gsdmd_33mer.pdb` |

The screen in step 3 ends close to the deposited pore, at a radius of about 127 Å and a tilt of about −21°. The parameters in step 4 are those that best reproduce the deposited 33-mer: Cα RMSD 0.6 Å after superposition, against 4.7 Å at best without `--gasdermin`.

## Example 2: *Vitiosangium* bacterial gasdermin pore (PDB 8SL0, 52-mer)

[`run_8sl0_bgsdm.sh`](run_8sl0_bgsdm.sh)

8SL0 is the cryo-EM structure of the bGSDM in a 'slinky'-like oligomer and contains a single protomer. It is only available as mmCIF, which `fetch_protomer.py` converts. The same publication describes a 52-mer pore, deposited as the integrative model [9A84](https://pdb-ihm.org/entry.html?9A84) in PDB-IHM.

| Step | Command | Output |
|---|---|---|
| 1 | `fetch_protomer.py 8SL0 A` | `8SL0.cif`, `bgsdm_protomer.pdb` (residues 6–233) |
| 2 | `barrel-align` | `bgsdm_aligned.pdb` |
| 3 | `barrel-optimize --n_subunits 52` | `round0_results.csv`, `round1_results.csv`, `optimized_ring_*.pdb` |
| 4 | `barrel-build --radius 179 --tilt_angle -17 --score` | `bgsdm_52mer.pdb` |

The parameters in step 4 reproduce the 9A84 pore model with a Cα RMSD of 1.5 Å, without `--gasdermin`; with it, the best fit is 5.6 Å.

The screen in step 3 instead ends at a radius of about 185 Å, 6 Å wider than 9A84 (Cα RMSD 6.3 Å). At the published radius the rigid, unminimized protomers overlap more (higher `fa_rep`), so the score favours a slightly wider ring. This is why the screen is only a coarse guide: the final parameters have to be chosen by evaluating the models against prior knowledge and experimental data.

## Checking a model

Every model should be evaluated manually. For example:

- fit it into the corresponding cryo-EM map as a rigid body (e.g. with `fitmap` in ChimeraX), or
- overlay it with a known assembly, and check stoichiometry, pore diameter and membrane insertion.

The maps for the examples are EMD-21160 (6VFE) and EMD-40570 (8SL0).

## Using your own structure

`fetch_protomer.py` accepts any PDB ID or a local PDB/mmCIF file and writes one chain as input for `barrel-align`:

```bash
python examples/fetch_protomer.py 6CB8 A protomer.pdb        # from the RCSB PDB
python examples/fetch_protomer.py my_model.cif B protomer.pdb  # local file
```

[`example_workflow.py`](example_workflow.py) runs the same workflow through the Python API.

## References

- **6VFE**: Xia, S. *et al.* Gasdermin D pore structure reveals preferential release of mature interleukin-1. *Nature* **593**, 607–611 (2021). https://doi.org/10.1038/s41586-021-03478-3
- **8SL0, 9A84**: Johnson, A.G. *et al.* Structure and assembly of a bacterial gasdermin pore. *Nature* **628**, 657–663 (2024). https://doi.org/10.1038/s41586-024-07216-3
