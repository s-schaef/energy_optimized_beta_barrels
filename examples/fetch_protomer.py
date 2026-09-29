#!/usr/bin/env python3
"""
Extract a single protomer (one protein chain) from a published structure.

Downloads the entry from the RCSB PDB (PDB format if available, otherwise
mmCIF) or reads a local PDB/mmCIF file, and writes one chain as chain A of a
new PDB file that can be passed to barrel-align. Only standard amino-acid
residues (no ligands, waters or modified residues) and the first alternate
location of each atom are kept. The downloaded entry is saved next to the
output file.

Usage:
    python fetch_protomer.py 6VFE A gsdmd_protomer.pdb
    python fetch_protomer.py 8SL0 A bgsdm_protomer.pdb
    python fetch_protomer.py my_structure.cif B protomer.pdb
"""

import os
import sys
import argparse
import urllib.error
import urllib.request
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

RCSB_URL = "https://files.rcsb.org/download/{}.{}"


def download_entry(pdb_id, out_dir):
    """Download a PDB entry from RCSB (PDB format, falling back to mmCIF)."""
    pdb_id = pdb_id.upper()
    for ext in ('pdb', 'cif'):
        path = os.path.join(out_dir, f"{pdb_id}.{ext}")
        if os.path.exists(path):
            print(f"Using previously downloaded {path}")
            return path
        try:
            urllib.request.urlretrieve(RCSB_URL.format(pdb_id, ext), path)
            print(f"Downloaded {pdb_id}.{ext} from RCSB to {path}")
            return path
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    raise ValueError(f"Entry {pdb_id} is not available from RCSB")


def load_structure(path):
    """Read a PDB or mmCIF file with Biopython."""
    if path.lower().endswith(('.cif', '.mmcif')):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(path), path)


class ProtomerSelect(Select):
    """Keep standard amino-acid residues and the first alternate location."""

    def accept_residue(self, residue):
        return residue.id[0] == ' ' and 'CA' in residue

    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() in ('A', '1')


def extract_protomer(source, chain_id, output_pdb):
    """
    Write one chain of a structure as a single-protomer PDB file.

    Parameters:
        source (str): PDB ID (downloaded from RCSB) or path to a PDB/mmCIF file
        chain_id (str): Chain to extract (author chain ID)
        output_pdb (str): Output PDB file; the chain is renamed to A

    Returns:
        str: Path to the downloaded or given structure file
    """
    out_dir = os.path.dirname(os.path.abspath(output_pdb))
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(source):
        structure_file = source
    else:
        structure_file = download_entry(source, out_dir)

    model = load_structure(structure_file)[0]
    if chain_id not in model:
        available = ', '.join(chain.id for chain in model)
        raise ValueError(f"Chain {chain_id} not found in {structure_file}. "
                         f"Available chains: {available}")

    chain = model[chain_id]
    chain.detach_parent()
    chain.id = 'A'
    protomer = Structure('protomer')
    protomer.add(Model(0))
    protomer[0].add(chain)

    io = PDBIO()
    io.set_structure(protomer)
    io.save(output_pdb, ProtomerSelect())

    residues = [res for res in chain if ProtomerSelect().accept_residue(res)]
    print(f"Wrote chain {chain_id} of {os.path.basename(structure_file)} "
          f"({len(residues)} residues, {residues[0].id[1]}-{residues[-1].id[1]}) "
          f"to {output_pdb}")
    return structure_file


def main():
    parser = argparse.ArgumentParser(
        description='Extract one protein chain from a PDB entry or local '
                    'PDB/mmCIF file as input for barrel-align.')
    parser.add_argument('source', help='PDB ID (e.g. 6VFE) or path to a PDB/mmCIF file')
    parser.add_argument('chain', help='Chain ID to extract (e.g. A)')
    parser.add_argument('output', help='Output PDB file for the protomer')
    args = parser.parse_args()

    try:
        extract_protomer(args.source, args.chain, args.output)
    except (ValueError, OSError) as e:
        sys.exit(f"Error: {e}")


if __name__ == '__main__':
    main()
