"""Generate minimal PDB files for testing without requiring real protein structures."""

import numpy as np


def make_beta_strand_atoms(start_resid, chain='A', start_atom=1,
                           origin=np.array([0.0, 0.0, 0.0]),
                           direction=np.array([0.0, 0.0, 1.0]),
                           n_residues=6):
    """
    Generate PDB ATOM lines for a minimal beta-strand-like backbone.

    Each residue gets N, CA, C, O atoms spaced along the given direction
    with ~3.5 A rise per residue (typical beta strand).
    """
    lines = []
    atom_id = start_atom
    direction = direction / np.linalg.norm(direction)
    # Perpendicular for slight zigzag
    perp = np.cross(direction, [1, 0, 0])
    if np.linalg.norm(perp) < 0.01:
        perp = np.cross(direction, [0, 1, 0])
    perp = perp / np.linalg.norm(perp)

    for i in range(n_residues):
        resid = start_resid + i
        base = origin + direction * 3.5 * i
        # Zigzag in beta strand
        zigzag = perp * 1.0 * ((-1) ** i)

        atoms = {
            'N':  base + zigzag + direction * 0.0,
            'CA': base + zigzag + direction * 1.5,
            'C':  base + zigzag + direction * 2.3,
            'O':  base + zigzag + direction * 2.3 + perp * 1.2,
        }

        for name, pos in atoms.items():
            lines.append(
                f"ATOM  {atom_id:5d} {name:<4s} ALA {chain}{resid:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           "
                f"{name[0]:>2s}  "
            )
            atom_id += 1

    return lines, atom_id


def make_simple_monomer_pdb(filepath, n_strands=4, strand_length=8, spacing=5.0):
    """
    Write a minimal PDB with several parallel beta-strand-like chains
    forming a sheet. Useful for testing alignment and ring building.

    Parameters:
        filepath: output PDB path
        n_strands: number of beta strands
        strand_length: residues per strand
        spacing: distance between strands in Angstroms
    """
    all_lines = []
    atom_id = 1
    resid = 1
    strand_dir = np.array([0.0, 0.0, 1.0])

    for s in range(n_strands):
        origin = np.array([0.0, spacing * s, 0.0])
        lines, atom_id = make_beta_strand_atoms(
            start_resid=resid, chain='A', start_atom=atom_id,
            origin=origin, direction=strand_dir, n_residues=strand_length
        )
        all_lines.extend(lines)
        resid += strand_length

    all_lines.append("END")

    with open(filepath, 'w') as f:
        f.write('\n'.join(all_lines) + '\n')

    return filepath


def make_helix_monomer_pdb(filepath, n_residues=30):
    """
    Write a minimal PDB with an alpha-helix-like backbone (no beta strands).
    Used to test the fallback alignment path.
    """
    lines = []
    atom_id = 1
    # Alpha helix: 1.5 A rise, 100 deg turn per residue
    for i in range(n_residues):
        resid = i + 1
        angle = np.radians(100 * i)
        r = 2.3  # helix radius
        z = 1.5 * i
        base = np.array([r * np.cos(angle), r * np.sin(angle), z])

        atoms = {
            'N':  base + np.array([0.0, 0.0, 0.0]),
            'CA': base + np.array([0.5, 0.0, 0.7]),
            'C':  base + np.array([1.0, 0.0, 1.2]),
            'O':  base + np.array([1.0, 1.0, 1.2]),
        }

        for name, pos in atoms.items():
            lines.append(
                f"ATOM  {atom_id:5d} {name:<4s} ALA A{resid:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           "
                f"{name[0]:>2s}  "
            )
            atom_id += 1

    lines.append("END")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return filepath
