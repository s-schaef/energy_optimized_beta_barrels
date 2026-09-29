# Changelog

## 1.0.0 (2026-09-29)

Release accompanying the publication. Changes relative to the scripts of
August 2025 (`alignment_module.py`, `ring_builder.py` and
`optimization_module.py` at commit `6e0871f`):

### Changed

- The code is an installable package (`barrel_builder`) with the command-line
  tools `barrel-align`, `barrel-build` and `barrel-optimize`. The old script
  names still work as thin wrappers.
- PyRosetta is only imported when scoring is requested.
- The screen reports the best geometry over all rounds (previously: the best
  of the last round).

Ring building itself is unchanged: rings built from the same aligned protomer
with the same parameters are identical to those of the previous version.

### Fixed

- `barrel-align` could return the mirror image of the input protein
  (D-amino acids), depending on the orientation of the input file. The
  alignment now always applies a proper rotation. Protomers aligned with the
  previous version should be checked for chirality.
- `--gasdermin` had no effect in `optimization_module.py`: it was never
  applied to the scored assemblies and, since August 2025, not to the written
  ring either. It is now applied throughout.
- The ring written by the screen uses the reported best parameters
  (previously the mean of the two best geometries of the last round).
- The screen stops with an error if PyRosetta is missing or every evaluation
  fails, instead of writing a ring from unscored geometries.
- Secondary structure is assigned on protein residues only, and multi-chain
  input is rejected with a clear message.
- Default output name of `barrel-align` for input paths containing directories.
- PyRosetta is initialized once per worker process instead of once per
  evaluation.
