"""Kabsch-aligned Calpha coordinate extraction for VAE input."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBParser, Superimposer

from af_claseq.utils.logging_utils import get_logger

logger = get_logger("umap_voting.coords")


def residue_range_from_sa(
    sa_json_path: str | Path, target: str
) -> List[int]:
    """Extract residue indices from a structure-analysis JSON config.

    Supports the real repo schema: basics.full_index, basics.local_index.
    Falls back to flat full_index / local_index for simplified test configs.
    """
    with open(sa_json_path) as f:
        sa = json.load(f)

    basics = sa.get("basics", sa)

    if target == "global":
        r = basics["full_index"]
        return list(range(int(r["start"]), int(r["end"]) + 1))

    local = basics.get("local_index")
    if local is None:
        raise ValueError(f"no local_index in {sa_json_path}")
    if isinstance(local, list):
        if not local:
            raise ValueError(f"local_index is an empty list in {sa_json_path}")
        local = local[0]
    return list(range(int(local["start"]), int(local["end"]) + 1))


def superposition_indices_from_sa(
    sa_json_path: str | Path,
) -> Optional[List[int]]:
    """Extract superposition indices from filter_criteria if available."""
    with open(sa_json_path) as f:
        sa = json.load(f)
    basics = sa.get("basics", sa)
    full = basics.get("full_index")
    if full:
        return list(range(int(full["start"]), int(full["end"]) + 1))
    return None


def _get_ca_coords_for_chain(
    pdb_path: str,
    chain_id: str,
    residue_indices: Sequence[int],
) -> Optional[np.ndarray]:
    """Extract Calpha coordinates for specified residues from a PDB chain."""
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("s", pdb_path)
    except Exception as e:
        logger.warning(f"PDB parse failed for {pdb_path}: {e}")
        return None

    model = struct[0]
    chain = None
    for c in model:
        if c.id == chain_id:
            chain = c
            break
    if chain is None:
        logger.warning(f"chain {chain_id} not found in {pdb_path}")
        return None

    target_set = set(residue_indices)
    coords = {}
    for res in chain:
        if "CA" not in res:
            continue
        resseq = res.get_id()[1]
        if resseq in target_set:
            coords[resseq] = res["CA"].get_coord()

    if len(coords) < len(residue_indices):
        return None

    return np.stack(
        [coords[i] for i in sorted(residue_indices)]
    ).astype(np.float32)


def extract_aligned_coords(
    pdb_path: str,
    chain_id: str,
    coord_indices: Sequence[int],
    superposition_indices: Sequence[int],
    alignment_ref_coords: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract Kabsch-aligned Calpha coordinates from a PDB file.

    If alignment_ref_coords is provided, the structure is superposed onto the
    reference over superposition_indices before extracting coord_indices.
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("s", pdb_path)
    except Exception as e:
        logger.warning(f"PDB parse failed for {pdb_path}: {e}")
        return None

    model = struct[0]
    chain = None
    for c in model:
        if c.id == chain_id:
            chain = c
            break
    if chain is None:
        logger.warning(f"chain {chain_id} not found in {pdb_path}")
        return None

    all_needed = set(coord_indices) | set(superposition_indices)
    ca_atoms = {}
    for res in chain:
        if "CA" not in res:
            continue
        resseq = res.get_id()[1]
        if resseq in all_needed:
            ca_atoms[resseq] = res["CA"]

    sup_atoms = [ca_atoms.get(i) for i in superposition_indices]
    if any(a is None for a in sup_atoms):
        return None

    if alignment_ref_coords is not None:
        from Bio.PDB.Atom import Atom

        ref_atoms = []
        for i, idx in enumerate(superposition_indices):
            a = Atom("CA", alignment_ref_coords[i], 0.0, 1.0, " ", "CA", i, "C")
            ref_atoms.append(a)

        sup = Superimposer()
        sup.set_atoms(ref_atoms, sup_atoms)
        sup.apply(struct.get_atoms())

    coord_atoms = [ca_atoms.get(i) for i in coord_indices]
    if any(a is None for a in coord_atoms):
        return None

    return np.stack([a.get_coord() for a in coord_atoms]).astype(np.float32)


class CoordExtractor:
    """Extracts aligned Calpha coordinate tensors from a set of PDB files."""

    def __init__(
        self,
        sa_json_path: str | Path,
        coord_target: str,
        alignment_ref_pdb: Optional[str] = None,
        alignment_ref_chain: str = "A",
        target_chain: str = "A",
    ):
        self.coord_indices = residue_range_from_sa(sa_json_path, coord_target)
        sup = superposition_indices_from_sa(sa_json_path)
        self.superposition_indices = sup if sup else self.coord_indices
        self.alignment_ref_chain = alignment_ref_chain
        self.target_chain = target_chain
        self.n_residues = len(self.coord_indices)

        self.alignment_ref_coords = None
        if alignment_ref_pdb:
            ref_coords = _get_ca_coords_for_chain(
                alignment_ref_pdb,
                alignment_ref_chain,
                self.superposition_indices,
            )
            if ref_coords is None:
                raise ValueError(
                    f"cannot extract alignment ref coords from {alignment_ref_pdb}"
                )
            self.alignment_ref_coords = ref_coords

    def extract(
        self, pdb_path: str, chain_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Extract aligned Calpha coords for a single PDB. Returns (L, 3) or None."""
        cid = chain_id if chain_id else self.target_chain
        return extract_aligned_coords(
            pdb_path,
            cid,
            self.coord_indices,
            self.superposition_indices,
            self.alignment_ref_coords,
        )
