import os
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio.PDB import PDBParser, Superimposer, PPBuilder
import json
from tqdm import tqdm

from af_claseq.utils.sequence_processing import count_sequences_in_a3m
from af_claseq.utils.logging_utils import get_logger

# Default path for TMalign executable
DEFAULT_TMALIGN_PATH = "/fs/ess/PAA0203/xing244/TMalign"

# Get logger for this module
logger = get_logger(__name__)


class StructureAnalyzer:
    """
    A class to perform structural analysis on PDB files, including angle calculation,
    residue distance measurement, RMSD computation, and pLDDT processing.
    """

    def __init__(self):
        """Initialize the StructureAnalyzer with a PDB parser."""
        self.pdb_parser = PDBParser(QUIET=True)

    def calculate_com(self, atoms: Sequence[Any]) -> np.ndarray:
        """
        Calculate the center of mass for a list of atoms.

        Args:
            atoms: Sequence of atom objects.

        Returns:
            Center of mass coordinates.

        Raises:
            ValueError: If no atoms provided.
        """
        if not atoms:
            raise ValueError("No atoms provided for center of mass calculation.")
            
        coordinates = np.array([atom.get_coord() for atom in atoms])
        masses = np.array([atom.mass for atom in atoms])
        return np.average(coordinates, axis=0, weights=masses)

    def calculate_angle(
        self,
        pdb_file: str | Path,
        domain1_indices: Sequence[int],
        domain2_indices: Sequence[int], 
        hinge_indices: Sequence[int]
    ) -> float:
        """
        Calculate the angle between two domains using their centers of mass and a hinge point.

        Args:
            pdb_file: Path to the PDB file.
            domain1_indices: Residue indices for domain 1.
            domain2_indices: Residue indices for domain 2. 
            hinge_indices: Residue indices for the hinge.

        Returns:
            Angle in degrees.

        Raises:
            Exception: If calculation fails.
        """
        try:
            structure = self.pdb_parser.get_structure("Protein", pdb_file)

            domain1_atoms = self.get_atoms_from_residue_indices(structure, domain1_indices)
            domain2_atoms = self.get_atoms_from_residue_indices(structure, domain2_indices)
            hinge_atoms = self.get_atoms_from_residue_indices(structure, hinge_indices)

            com_domain1 = self.calculate_com(domain1_atoms)
            com_domain2 = self.calculate_com(domain2_atoms)
            com_hinge = self.calculate_com(hinge_atoms)

            v1 = com_domain1 - com_hinge
            v2 = com_domain2 - com_hinge
            
            # Normalize vectors for more stable calculation
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            cos_angle = np.dot(v1_norm, v2_norm)
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle_rad)

        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            raise

    def calculate_residue_distance(
        self,
        pdb_file: str | Path,
        chain_id: str,
        residue_indices1: Sequence[int],
        residue_indices2: Sequence[int]
    ) -> Optional[float]:
        """
        Calculate the distance between two sets of residues in a specific chain.

        Args:
            pdb_file: Path to the PDB file.
            chain_id: Chain identifier.
            residue_indices1: Residue indices for the first set.
            residue_indices2: Residue indices for the second set.

        Returns:
            Distance between residue centers or None if calculation fails.
        """
        try:
            structure = self.pdb_parser.get_structure("Protein", pdb_file)
            center1 = self.get_residue_center(structure, chain_id, residue_indices1)
            center2 = self.get_residue_center(structure, chain_id, residue_indices2)

            if center1 is not None and center2 is not None:
                return float(np.linalg.norm(center1 - center2))
            return None
            
        except Exception as e:
            logger.error(f"Error calculating residue distance: {e}")
            return None

    def calculate_ca_rmsd(
        self,
        reference_pdb: str | Path,
        target_pdb: str | Path,
        superposition_indices: Sequence[int],
        rmsd_indices: Sequence[int],
        chain_id: str = "A"
    ) -> float:
        """
        Calculate the RMSD of CA atoms between reference and target PDB for specified residues.
        The residue sets for superimposition may or may not be the same as the RMSD calcauting residue indices

        Args:
            reference_pdb: Path to the reference PDB file.
            target_pdb: Path to the target PDB file.
            superposition_indices: Residue indices used for superposition.
            rmsd_indices: Residue indices used for RMSD calculation.
            chain_id: Chain identifier. Defaults to "A".

        Returns:
            RMSD value.

        Raises:
            Exception: If RMSD calculation fails.
        """
        try:
            ref_structure = self.pdb_parser.get_structure("reference", reference_pdb)
            target_structure = self.pdb_parser.get_structure("target", target_pdb)

            def get_ca_atoms(structure, indices):
                atoms = []
                for res_id in indices:
                    try:
                        res = structure[0][chain_id][res_id]
                        atoms.append(res["CA"])
                    except KeyError:
                        logger.warning(f"Residue {res_id} not found. Skipping.")
                return atoms

            ref_sup_atoms = get_ca_atoms(ref_structure, superposition_indices)
            target_sup_atoms = get_ca_atoms(target_structure, superposition_indices)

            if not ref_sup_atoms or not target_sup_atoms:
                logger.warning("No CA atoms found for superposition.")
                return float('nan')

            # Perform superposition
            super_imposer = Superimposer()
            super_imposer.set_atoms(ref_sup_atoms, target_sup_atoms)
            super_imposer.apply(target_structure.get_atoms())

            ref_rmsd_atoms = get_ca_atoms(ref_structure, rmsd_indices)
            target_rmsd_atoms = get_ca_atoms(target_structure, rmsd_indices)

            if not ref_rmsd_atoms or not target_rmsd_atoms:
                logger.warning("No CA atoms found for RMSD calculation.")
                return float('nan')

            return self._calculate_rmsd(ref_rmsd_atoms, target_rmsd_atoms)

        except Exception as e:
            logger.error(f"Error calculating CA RMSD: {e}")
            raise

    def _calculate_rmsd(self, atoms1: Sequence[Any], atoms2: Sequence[Any]) -> float:
        """Helper method to calculate RMSD between two lists of atoms."""
        if len(atoms1) != len(atoms2):
            raise ValueError("Atom lists must have same length")
        
        coords1 = np.array([a.get_coord() for a in atoms1])
        coords2 = np.array([a.get_coord() for a in atoms2])
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    
    def calculate_all_atom_rmsd(
        self,
        reference_pdb: str | Path,
        target_pdb: str | Path,
        superposition_indices: Sequence[int],
        rmsd_indices: Sequence[int],
        chain_id: str = "A"
    ) -> float:
        """
        Calculate RMSD between two structures using all atoms in specified residues.
        
        Args:
            reference_pdb: Path to the reference PDB file.
            target_pdb: Path to the target PDB file.
            superposition_indices: Residue indices used for superposition.
            rmsd_indices: Residue indices used for RMSD calculation.
            chain_id: Chain identifier. Defaults to "A".

        Returns:
            RMSD value.

        Raises:
            Exception: If RMSD calculation fails.
        """
        try:
            ref_structure = self.pdb_parser.get_structure("reference", reference_pdb)
            target_structure = self.pdb_parser.get_structure("target", target_pdb)

            def get_ca_atoms(structure, indices):
                atoms = []
                for res_id in indices:
                    try:
                        res = structure[0][chain_id][res_id]
                        atoms.append(res["CA"])
                    except KeyError:
                        logger.warning(f"Residue {res_id} not found. Skipping.")
                return atoms
            
            def get_all_atoms(structure, indices):
                atoms = []
                for res_id in indices:
                    try:
                        res = structure[0][chain_id][res_id]
                        for atom in res:
                            atoms.append(atom)
                    except KeyError:
                        logger.warning(f"Residue {res_id} not found. Skipping.")
                return atoms

            # Use CA atoms for superposition
            ref_sup_atoms = get_ca_atoms(ref_structure, superposition_indices)
            target_sup_atoms = get_ca_atoms(target_structure, superposition_indices)

            if not ref_sup_atoms or not target_sup_atoms:
                logger.warning("No CA atoms found for superposition.")
                return float('nan')

            # Perform superposition
            super_imposer = Superimposer()
            super_imposer.set_atoms(ref_sup_atoms, target_sup_atoms)
            super_imposer.apply(target_structure.get_atoms())

            # Get all atoms for RMSD calculation
            ref_all_atoms = get_all_atoms(ref_structure, rmsd_indices)
            target_all_atoms = get_all_atoms(target_structure, rmsd_indices)
            
            if not ref_all_atoms or not target_all_atoms:
                logger.warning("No atoms found for RMSD calculation.")
                return float('nan')
            
            # Match atoms between reference and target
            matched_ref_atoms = []
            matched_target_atoms = []
            
            # Create a dictionary of target atoms by name and residue
            target_atom_dict = {}
            for atom in target_all_atoms:
                res_id = atom.get_parent().id[1]
                atom_name = atom.get_name()
                target_atom_dict[(res_id, atom_name)] = atom
            
            # Match reference atoms with target atoms
            for atom in ref_all_atoms:
                res_id = atom.get_parent().id[1]
                atom_name = atom.get_name()
                key = (res_id, atom_name)
                
                if key in target_atom_dict:
                    matched_ref_atoms.append(atom)
                    matched_target_atoms.append(target_atom_dict[key])
            
            if not matched_ref_atoms or not matched_target_atoms:
                logger.warning("No matching atoms found for RMSD calculation.")
                return float('nan')
                
            return self._calculate_rmsd(matched_ref_atoms, matched_target_atoms)

        except Exception as e:
            logger.error(f"Error calculating all-atom RMSD: {e}")
            raise

    def calculate_tm_score(
        self,
        target_pdb: str | Path,
        reference_pdb: str | Path,
        tm_align_path: str | Path = DEFAULT_TMALIGN_PATH
    ) -> float:
        """
        Calculate TM-score between target and reference PDB structures.

        Args:
            target_pdb: Path to the target PDB file.
            reference_pdb: Path to the reference PDB file.
            tm_align_path: Path to the TMalign executable.

        Returns:
            TM-score value.

        Raises:
            Exception: If TM-score calculation fails.
        """
        try:
            result = subprocess.run(
                [str(tm_align_path), str(target_pdb), str(reference_pdb)],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if "if normalized by length of Chain_2" in line:
                    return float(line.split()[1])
                    
            raise ValueError("Could not find TM-score in TMalign output")
            
        except Exception as e:
            logger.error(f"Error calculating TM-score: {e}")
            raise

    def plddt_process(self,
                     pdb_file_path: str | Path,
                     residue_indices: Sequence[int]) -> Optional[float]:
        """
        Calculate the average pLDDT score for specified residues in a PDB file.

        Args:
            pdb_file_path: Path to the PDB file.
            residue_indices: Residue indices to consider.

        Returns:
            Average pLDDT score or None if not available.
        """
        try:
            b_factors = []
            residue_set = set(residue_indices)  # For faster lookup
            
            with open(pdb_file_path) as pdb_file:
                for line in pdb_file:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        residue_index = int(line[22:26])
                        if residue_index in residue_set:
                            b_factors.append(float(line[60:66]))

            return float(np.mean(b_factors)) if b_factors else None
            
        except Exception as e:
            logger.error(f"Error processing pLDDT for {pdb_file_path}: {e}")
            return None

    def get_residue_center(
        self, 
        structure: Any, 
        chain_id: str, 
        residue_indices: Sequence[int]
    ) -> Optional[np.ndarray]:
        """
        Calculate the center of mass for a set of residues in a specific chain.

        Args:
            structure: Parsed PDB structure.
            chain_id: Chain identifier.
            residue_indices: Residue indices.

        Returns:
            Center of mass coordinates or None if residues not found.
        """
        try:
            atoms = []
            for res_index in residue_indices:
                try:
                    residue = structure[0][chain_id][res_index]
                    atoms.extend(list(residue.get_atoms()))
                except KeyError:
                    logger.error(f"Residue {res_index} not found in chain {chain_id}")
                    return None
                    
            return self.calculate_com(atoms) if atoms else None
            
        except Exception as e:
            logger.error(f"Error calculating residue center: {e}")
            return None

    def get_atoms_from_residue_indices(
        self, 
        structure: Any, 
        residue_indices: Sequence[int]
    ) -> List[Any]:
        """
        Retrieve all atoms from specified residue indices in a structure.

        Args:
            structure: Parsed PDB structure.
            residue_indices: Residue indices.

        Returns:
            List of atom objects.
        """
        residue_set = set(residue_indices)
        atoms = []
        for residue in structure.get_residues():
            if residue.get_id()[1] in residue_set:
                atoms.extend(list(residue.get_atoms()))
        return atoms

    def process_single_pdb(self, 
                          pdb_path: str, 
                          filter_criteria: List[Dict[str, Any]], 
                          basics: Dict[str, Any], 
                          plddt_threshold: float = 0) -> Optional[Dict[str, Any]]:
        """
        Process a single PDB file and extract metrics based on filter criteria.
        
        Args:
            pdb_path: Path to the PDB file
            filter_criteria: List of criteria for filtering structures
            basics: Basic configuration parameters
            plddt_threshold: Minimum pLDDT score threshold
            
        Returns:
            Dictionary of metrics or None if processing fails or pLDDT below threshold
        """
        try:
            # Extract full index range for pLDDT calculation
            full_indices = self._extract_indices_from_spec(basics.get('full_index'))
            if not full_indices:
                logger.warning(f"No valid indices defined for pLDDT calculation in {pdb_path}")
                return None
                
            plddt = self.plddt_process(pdb_path, full_indices)
            
            if not plddt:
                logger.warning(f"Could not calculate pLDDT for {pdb_path}")
                return None
                
            if plddt <= plddt_threshold:
                return None
                
            # Initialize results with pLDDT
            result = {'PDB': pdb_path, 'plddt': plddt}
            
            # Get local pLDDT if specified
            if 'local_index' in basics:
                local_indices = self._extract_indices_from_spec(basics['local_index'])
                if local_indices:
                    result['local_plddt'] = self.plddt_process(pdb_path, local_indices)
            
            # Get sequence count if needed
            a3m_file = pdb_path.split("_unrelaxed")[0] + '.a3m'
            if os.path.exists(a3m_file):
                result['seq_count'] = count_sequences_in_a3m(a3m_file)
            
            # Process each criterion
            for criterion in filter_criteria:
                metric_name = criterion['name']
                criterion_type = criterion['type']
                
                if criterion_type == 'distance':
                    distance = self.calculate_residue_distance(
                        pdb_path, 'A', 
                        criterion['indices']['set1'],
                        criterion['indices']['set2']
                    )
                    result[metric_name] = distance
                    
                elif criterion_type == 'angle':
                    angle = self.calculate_angle(
                        pdb_path,
                        criterion['indices']['domain1'],
                        criterion['indices']['domain2'],
                        criterion['indices']['hinge']
                    )
                    result[metric_name] = angle
                    
                elif criterion_type in ('rmsd', 'all_atom_rmsd'):
                    sup_indices = self._extract_indices_from_spec(
                        criterion.get('superposition_indices'),
                        full_indices
                    )
                    rmsd_indices = self._extract_indices_from_spec(
                        criterion.get('rmsd_indices'),
                        full_indices
                    )
                    
                    # Use all-atom RMSD if specified
                    if criterion_type == 'all_atom_rmsd' or ('method' in criterion and criterion['method'] == 'all_atom_rmsd'):
                        rmsd = self.calculate_all_atom_rmsd(
                            criterion['ref_pdb'],
                            pdb_path,
                            sup_indices,
                            rmsd_indices,
                            chain_id='A'
                        )
                    else:
                        rmsd = self.calculate_ca_rmsd(
                            criterion['ref_pdb'],
                            pdb_path,
                            sup_indices,
                            rmsd_indices,
                            chain_id='A'
                        )
                    
                    result[metric_name] = rmsd
                    
                elif criterion_type == 'tmscore':
                    tm_score = self.calculate_tm_score(
                        pdb_path, 
                        criterion['ref_pdb']
                    )
                    result[metric_name] = tm_score

            return result
        except Exception as e:
            logger.error(f"Error processing {pdb_path}: {str(e)}")
            return None
    
    def _extract_indices_from_spec(self, indices_spec, default_indices=None):
        """Extract indices from specification or use defaults."""
        if not indices_spec:
            return default_indices
            
        if isinstance(indices_spec, list):
            result = []
            for range_dict in indices_spec:
                result.extend(range(range_dict['start'], range_dict['end']+1))
            return result
        else:
            return list(range(indices_spec['start'], indices_spec['end'] + 1))
        
    def get_result_df(self, parent_dir: str | Path,
                     filter_criteria: Sequence[Dict[str, Any]],
                     basics: Dict[str, Any],
                     plddt_threshold: float = 0) -> pd.DataFrame:
        """
        Generate a DataFrame containing calculated properties for each PDB file.

        Args:
            parent_dir: Path to the parent directory containing PDB files.
            filter_criteria: List of criteria for filtering and calculation.
            basics: Basic information required for calculations.
            plddt_threshold: Minimum pLDDT score threshold

        Returns:
            DataFrame with results.
        """
        logger.info(f'Processing {parent_dir}')

        # Use pathlib for better path handling
        parent_path = Path(parent_dir)
        pdb_files = [
            str(f) for f in parent_path.rglob('*.pdb')
            if 'non_a3m' not in str(f.parent)
        ]
        logger.info(f'Found {len(pdb_files)} PDB files')

        analyzer = StructureAnalyzer()

        # Process PDB files in parallel
        results = Parallel(n_jobs=-1)(
            delayed(analyzer.process_single_pdb)(
                pdb, filter_criteria, basics, plddt_threshold
            ) for pdb in tqdm(pdb_files, desc="Processing PDB files")
        )

        # Filter out None results
        results = [r for r in results if r is not None]

        return pd.DataFrame(results)



def get_protein_sequence(pdb_filename: str | Path) -> str:
    """
    Extract the protein sequence from a PDB file.

    Args:
        pdb_filename: Path to the PDB file.

    Returns:
        The full protein sequence.

    Raises:
        Exception: If there's an error in processing the PDB file.
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("Protein", pdb_filename)
        return ''.join(str(pp.get_sequence()) for pp in PPBuilder().build_peptides(structure))
    except Exception as e:
        logger.error(f"Error getting protein sequence: {e}")
        raise

def load_filter_modes(file_path: str | Path) -> Dict:
    """
    Load filter modes from a JSON file.

    Args:
        file_path: Path to the JSON file containing filter modes.

    Returns:
        The loaded filter modes.
    """
    with open(file_path) as f:
        return json.load(f)

def apply_filters(
    df_threshold: pd.DataFrame,
    df_operate: pd.DataFrame,
    filter_criteria: Sequence[Dict[str, Any]],
    quantile: float
) -> pd.DataFrame:
    """
    Apply filters to the dataframe based on specified criteria and quantile thresholds.

    Args:
        df_threshold: DataFrame used for calculating thresholds.
        df_operate: DataFrame to apply filters on.
        filter_criteria: List of filter criteria dictionaries.
        quantile: Quantile value for threshold calculation.

    Returns:
        Filtered DataFrame.
    """
    filtered_df = df_operate.copy()
    
    for criterion in filter_criteria:
        column_name = criterion['name']
        method = criterion['method']
        
        if method not in {'above', 'below'}:
            logger.warning(f"Invalid filter method: {method}. Skipping this criterion.")
            continue
            
        if column_name not in df_threshold.columns:
            logger.warning(f"Column {column_name} not found in threshold DataFrame. Skipping.")
            continue
            
        threshold_value = df_threshold[column_name].quantile(1 - quantile if method == 'above' else quantile)
        mask = filtered_df[column_name] > threshold_value if method == 'above' else filtered_df[column_name] < threshold_value
        filtered_df = filtered_df[mask]
        
        logger.info(
            f"Filtering {column_name} {method} {(1-quantile if method == 'above' else quantile)*100}% "
            f"quantile value: {threshold_value}"
        )
    
    return filtered_df
