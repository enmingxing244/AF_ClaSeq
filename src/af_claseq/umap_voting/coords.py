"""Kabsch-aligned Calpha coordinate extraction for VAE input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from Bio.PDB import PDBParser

from af_claseq.utils.logging_utils import get_logger

from .config import CoordExtractionSection, StructureAnalysisSection

logger = get_logger("umap_voting.coords")

_parser = PDBParser(QUIET=True)


# ---------------------------------------------------------------------------
# Index specification parsing
# ---------------------------------------------------------------------------

def _parse_index_spec(spec) -> List[int]:
    """Parse flexible index specifications into a flat sorted list of ints.

    Accepted forms:
      - ``{"start": 1, "end": 10}`` → [1, 2, ..., 10]
      - ``[{"start": 1, "end": 5}, {"start": 8, "end": 10}]`` → discontinuous
      - bare ``int`` → single-element list
    """
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, dict):
        return list(range(int(spec["start"]), int(spec["end"]) + 1))
    if isinstance(spec, list):
        indices: List[int] = []
        for item in spec:
            indices.extend(_parse_index_spec(item))
        return sorted(set(indices))
    raise ValueError(f"Cannot parse index spec: {spec!r}")


# ---------------------------------------------------------------------------
# Structure-analysis JSON helpers
# ---------------------------------------------------------------------------

def residue_range_from_sa(config_json: str | Path, target: str) -> List[int]:
    """Extract residue indices from a structure-analysis JSON.

    ``target`` is ``"local"`` or ``"global"``, mapping to ``local_index`` or
    ``full_index`` in the JSON ``basics`` section.
    """
    with open(config_json) as f:
        sa = json.load(f)
    basics = sa["basics"]
    key = "local_index" if target == "local" else "full_index"
    if key not in basics:
        raise KeyError(
            f"Structure-analysis JSON has no '{key}' in basics "
            f"(available: {list(basics.keys())})"
        )
    return _parse_index_spec(basics[key])


def superposition_indices_from_sa(config_json: str | Path) -> List[int]:
    """Get superposition indices (full_index) from a structure-analysis JSON."""
    with open(config_json) as f:
        sa = json.load(f)
    return _parse_index_spec(sa["basics"]["full_index"])


# ---------------------------------------------------------------------------
# Low-level CA extraction
# ---------------------------------------------------------------------------

def _get_ca_coords_for_chain(
    pdb_path: str | Path,
    chain_id: str,
    residue_indices: Sequence[int],
    strict: bool = True,
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """Extract CA coordinates for *residue_indices* from a PDB chain.

    When *strict* is True (default), returns an ordered ``(N, 3)`` array and
    raises if any residue is missing.  When False, returns a dict
    ``{resid: coord}`` and silently skips missing residues.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    model = structure[0]
    if chain_id not in model:
        raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}")
    chain = model[chain_id]

    if strict:
        coords = []
        for rid in residue_indices:
            try:
                coords.append(chain[rid]["CA"].get_vector().get_array())
            except KeyError:
                raise ValueError(
                    f"Residue {rid} or its CA missing in {pdb_path} chain {chain_id}"
                )
        return np.array(coords, dtype=np.float32)
    else:
        result: Dict[int, np.ndarray] = {}
        for rid in residue_indices:
            try:
                result[rid] = chain[rid]["CA"].get_vector().get_array().astype(np.float32)
            except KeyError:
                pass
        return result


# ---------------------------------------------------------------------------
# Kabsch alignment + extraction
# ---------------------------------------------------------------------------

def extract_aligned_coords(
    pdb_path: str | Path,
    chain_id: str,
    residue_indices: Sequence[int],
    superposition_indices: Sequence[int],
    alignment_ref_coords: Optional[Union[np.ndarray, Dict[int, np.ndarray]]] = None,
    min_superposition_atoms: int = 30,
) -> Optional[np.ndarray]:
    """Kabsch-align a structure and return target residue CA coordinates.

    When *alignment_ref_coords* is a dict (from ``strict=False``), finds
    common residues between the structure and the reference for superposition.
    """
    if alignment_ref_coords is None:
        return _get_ca_coords_for_chain(pdb_path, chain_id, residue_indices, strict=True)

    # Get superposition atoms from structure (non-strict → dict)
    struct_sup = _get_ca_coords_for_chain(
        pdb_path, chain_id, superposition_indices, strict=False
    )

    if isinstance(alignment_ref_coords, dict):
        # Common-residue superposition
        common = sorted(set(struct_sup.keys()) & set(alignment_ref_coords.keys()))
        if len(common) < min_superposition_atoms:
            logger.warning(
                f"{pdb_path}: only {len(common)} common superposition atoms "
                f"(need {min_superposition_atoms})"
            )
            if len(common) < 3:
                return None
        ref_arr = np.array([alignment_ref_coords[r] for r in common], dtype=np.float64)
        mov_arr = np.array([struct_sup[r] for r in common], dtype=np.float64)
    else:
        # Ordered array superposition
        mov_arr_raw = _get_ca_coords_for_chain(
            pdb_path, chain_id, superposition_indices, strict=False
        )
        if isinstance(mov_arr_raw, dict):
            ordered = sorted(mov_arr_raw.keys())
            mov_arr = np.array([mov_arr_raw[k] for k in ordered], dtype=np.float64)
        else:
            mov_arr = mov_arr_raw.astype(np.float64)
        ref_arr = alignment_ref_coords[:len(mov_arr)].astype(np.float64)

    # Kabsch via SVD (avoids needing BioPython Atom objects)
    ref_center = ref_arr.mean(axis=0)
    mov_center = mov_arr.mean(axis=0)
    ref_c = ref_arr - ref_center
    mov_c = mov_arr - mov_center

    H = mov_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])
    rot = Vt.T @ sign_matrix @ U.T  # R maps mov onto ref
    tran = ref_center - mov_center @ rot

    # Extract target residues (strict)
    try:
        target_coords = _get_ca_coords_for_chain(
            pdb_path, chain_id, residue_indices, strict=True
        )
    except ValueError:
        return None

    aligned = np.dot(target_coords.astype(np.float64), rot) + tran
    return aligned.astype(np.float32)


# ---------------------------------------------------------------------------
# Stateful extractor
# ---------------------------------------------------------------------------

class CoordExtractor:
    """Caches alignment reference and extraction parameters."""

    def __init__(
        self,
        sa_section: StructureAnalysisSection,
        coord_section: CoordExtractionSection,
    ):
        self.residue_indices = residue_range_from_sa(
            sa_section.config_json, sa_section.coord_target
        )
        self.superposition_indices = superposition_indices_from_sa(sa_section.config_json)
        self.chain_id = coord_section.target_chain
        self.min_superposition_atoms = coord_section.min_superposition_atoms

        self.alignment_ref_coords: Optional[Union[np.ndarray, Dict[int, np.ndarray]]] = None
        if coord_section.alignment_ref_pdb is not None:
            self.alignment_ref_coords = _get_ca_coords_for_chain(
                coord_section.alignment_ref_pdb,
                coord_section.alignment_ref_chain,
                self.superposition_indices,
                strict=False,
            )

        logger.info(
            f"CoordExtractor: {len(self.residue_indices)} target residues, "
            f"{len(self.superposition_indices)} superposition residues"
        )

    def extract(self, pdb_path: str | Path) -> Optional[np.ndarray]:
        """Return ``(L, 3)`` float32 array or None on failure."""
        try:
            return extract_aligned_coords(
                pdb_path,
                self.chain_id,
                self.residue_indices,
                self.superposition_indices,
                self.alignment_ref_coords,
                self.min_superposition_atoms,
            )
        except Exception as e:
            logger.warning(f"Extraction failed for {pdb_path}: {e}")
            return None
