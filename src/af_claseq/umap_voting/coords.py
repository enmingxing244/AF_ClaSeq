"""Kabsch-aligned Calpha coordinate extraction for VAE input."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from Bio.PDB import PDBParser, Superimposer

from af_claseq.utils.logging_utils import get_logger

logger = get_logger("umap_voting.coords")


def _parse_index_spec(spec) -> List[int]:
    """Parse an index specification into a flat list of ints.

    Accepts a single ``{start, end}`` dict, a list of such dicts
    (discontinuous ranges), or a bare int / list of ints.  Consistent
    with ``StructureAnalyzer._extract_indices_from_spec`` and
    ``scatter._parse_indices``.
    """
    if isinstance(spec, dict):
        return list(range(int(spec["start"]), int(spec["end"]) + 1))
    if isinstance(spec, list):
        out: List[int] = []
        for item in spec:
            if isinstance(item, dict):
                out.extend(range(int(item["start"]), int(item["end"]) + 1))
            else:
                out.append(int(item))
        return out
    return [int(spec)]


def residue_range_from_sa(
    sa_json_path: str | Path, target: str
) -> List[int]:
    """Extract residue indices from a structure-analysis JSON config.

    Supports the real repo schema: basics.full_index, basics.local_index.
    Both may be a single ``{start, end}`` dict or a list of dicts for
    discontinuous ranges.
    """
    with open(sa_json_path) as f:
        sa = json.load(f)

    basics = sa.get("basics", sa)

    if target == "global":
        r = basics["full_index"]
        return _parse_index_spec(r)

    local = basics.get("local_index")
    if local is None:
        raise ValueError(f"no local_index in {sa_json_path}")
    return _parse_index_spec(local)


def superposition_indices_from_sa(
    sa_json_path: str | Path,
) -> Optional[List[int]]:
    """Extract superposition indices from basics.full_index."""
    with open(sa_json_path) as f:
        sa = json.load(f)
    basics = sa.get("basics", sa)
    full = basics.get("full_index")
    if full:
        return _parse_index_spec(full)
    return None


DEFAULT_MIN_SUPERPOSITION_ATOMS = 30


def _get_ca_coords_for_chain(
    pdb_path: str,
    chain_id: str,
    residue_indices: Sequence[int],
    strict: bool = True,
) -> Optional[Union[np.ndarray, Dict[int, np.ndarray]]]:
    """Extract Calpha coordinates for specified residues from a PDB chain.

    When *strict* is True (default) all residues must be present or None
    is returned (as an ordered ndarray).  When False, only the residues
    that exist are returned as a dict ``{resid: coord}`` and the caller
    decides what to do with missing entries.
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

    target_set = set(residue_indices)
    coords = {}
    for res in chain:
        if "CA" not in res:
            continue
        resseq = res.get_id()[1]
        if resseq in target_set:
            coords[resseq] = res["CA"].get_coord()

    if not strict:
        return coords

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
    alignment_ref_coords: Optional[dict] = None,
    min_superposition_atoms: int = DEFAULT_MIN_SUPERPOSITION_ATOMS,
) -> Optional[np.ndarray]:
    """Extract Kabsch-aligned Calpha coordinates from a PDB file.

    Superposition uses the **common** residues between the structure and
    the reference (like ``StructureAnalyzer.calculate_ca_rmsd``).  This
    handles references with gaps or structures shorter than
    *superposition_indices* gracefully.

    *alignment_ref_coords* is a ``{resid: xyz_array}`` dict (not an
    ordered array) so the caller need not know which residues the
    reference contains upfront.

    The coord_indices extraction remains strict: all requested residues
    must be present or None is returned.
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

    if alignment_ref_coords is not None:
        from Bio.PDB.Atom import Atom

        common = sorted(
            set(superposition_indices)
            & set(ca_atoms.keys())
            & set(alignment_ref_coords.keys())
        )
        if len(common) < min_superposition_atoms:
            logger.warning(
                f"{pdb_path}: only {len(common)} common superposition "
                f"atoms (need {min_superposition_atoms})"
            )
            return None

        ref_atoms = [
            Atom("CA", alignment_ref_coords[r], 0.0, 1.0, " ", "CA", i, "C")
            for i, r in enumerate(common)
        ]
        target_atoms = [ca_atoms[r] for r in common]

        sup = Superimposer()
        sup.set_atoms(ref_atoms, target_atoms)
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
        min_superposition_atoms: int = DEFAULT_MIN_SUPERPOSITION_ATOMS,
    ):
        self.coord_indices = residue_range_from_sa(sa_json_path, coord_target)
        sup = superposition_indices_from_sa(sa_json_path)
        self.superposition_indices = sup if sup else self.coord_indices
        self.alignment_ref_chain = alignment_ref_chain
        self.target_chain = target_chain
        self.n_residues = len(self.coord_indices)
        self.min_superposition_atoms = min_superposition_atoms

        self.alignment_ref_coords = None
        if alignment_ref_pdb:
            ref_coords = _get_ca_coords_for_chain(
                alignment_ref_pdb,
                alignment_ref_chain,
                self.superposition_indices,
                strict=False,
            )
            if not ref_coords:
                raise ValueError(
                    f"cannot extract alignment ref coords from {alignment_ref_pdb}"
                )
            n_avail = len(ref_coords)
            n_req = len(self.superposition_indices)
            if n_avail < n_req:
                logger.info(
                    f"alignment ref {alignment_ref_pdb}: {n_avail}/{n_req} "
                    f"superposition residues available"
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
            self.min_superposition_atoms,
        )
