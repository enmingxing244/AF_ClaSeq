import os
import logging
import subprocess
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio.PDB import PDBParser, Superimposer, PPBuilder
import json
from tqdm import tqdm

from af_path.sequence_processing import count_sequences_in_a3m

# Default path for TMalign executable
DEFAULT_TMALIGN_PATH = "/fs/ess/PAA0203/xing244/TMalign"


class StructureAnalyzer:
    """
    A class to perform structural analysis on PDB files, including angle calculation,
    residue distance measurement, RMSD computation, and pLDDT processing.
    """

    def __init__(self):
        """
        Initialize the StructureAnalyzer with a PDB parser.
        """
        self.pdb_parser = PDBParser(QUIET=True)

    def calculate_com(self, atoms: List[Any]) -> np.ndarray:
        """
        Calculate the center of mass for a list of atoms.

        Args:
            atoms (List[Any]): List of atom objects.

        Returns:
            np.ndarray: Center of mass coordinates.
        """
        coordinates = np.array([atom.get_coord() for atom in atoms])
        masses = np.array([atom.mass for atom in atoms])
        if len(masses) == 0:
            raise ValueError("No atoms provided for center of mass calculation.")
        com = np.average(coordinates, axis=0, weights=masses)
        return com

    def calculate_angle(
        self,
        pdb_file: str,
        domain1_indices: List[int],
        domain2_indices: List[int],
        hinge_indices: List[int],
    ) -> float:
        """
        Calculate the angle between two domains using their centers of mass and a hinge point.

        Args:
            pdb_file (str): Path to the PDB file.
            domain1_indices (List[int]): Residue indices for domain 1.
            domain2_indices (List[int]): Residue indices for domain 2.
            hinge_indices (List[int]): Residue indices for the hinge.

        Returns:
            float: Angle in degrees.

        Raises:
            Exception: If an error occurs during calculation.
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
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            return angle_deg
        except Exception as e:
            logging.error(f"Error calculating angle: {e}")
            raise

    def calculate_residue_distance(
        self,
        pdb_file: str,
        chain_id: str,
        residue_indices1: List[int],
        residue_indices2: List[int],
    ) -> Optional[float]:
        """
        Calculate the distance between two sets of residues in a specific chain.

        Args:
            pdb_file (str): Path to the PDB file.
            chain_id (str): Chain identifier.
            residue_indices1 (List[int]): Residue indices for the first set.
            residue_indices2 (List[int]): Residue indices for the second set.

        Returns:
            Optional[float]: Distance between residue centers or None if calculation fails.
        """
        try:
            structure = self.pdb_parser.get_structure("Protein", pdb_file)
            center1 = self.get_residue_center(structure, chain_id, residue_indices1)
            center2 = self.get_residue_center(structure, chain_id, residue_indices2)

            if center1 is not None and center2 is not None:
                distance = np.linalg.norm(center1 - center2)
                return distance
            else:
                return None
        except Exception as e:
            logging.error(f"Error calculating residue distance: {e}")
            return None

    def calculate_ca_rmsd(
        self,
        reference_pdb: str,
        target_pdb: str,
        superposition_indices: List[int],
        rmsd_indices: List[int],
        chain_id: str = "A",
    ) -> float:
        """
        Calculate the RMSD of CA atoms between a reference PDB and a target PDB for specified residues,
        using a separate set of residues for superposition.

        Args:
            reference_pdb (str): Path to the reference PDB file.
            target_pdb (str): Path to the target PDB file.
            superposition_indices (List[int]): Residue indices used for superposition.
            rmsd_indices (List[int]): Residue indices used for RMSD calculation.
            chain_id (str, optional): Chain identifier. Defaults to "A".

        Returns:
            float: RMSD value.

        Raises:
            Exception: If an error occurs during RMSD calculation.
        """
        try:
            ref_structure = self.pdb_parser.get_structure("reference", reference_pdb)
            target_structure = self.pdb_parser.get_structure("target", target_pdb)

            # Get atoms for superposition
            ref_sup_atoms = []
            target_sup_atoms = []
            for res_id in superposition_indices:
                try:
                    ref_res = ref_structure[0][chain_id][res_id]
                    target_res = target_structure[0][chain_id][res_id]
                    ref_sup_atoms.append(ref_res["CA"])
                    target_sup_atoms.append(target_res["CA"])
                except KeyError:
                    logging.warning(f"Residue {res_id} not found for superposition. Skipping.")

            if not ref_sup_atoms or not target_sup_atoms:
                logging.warning("No CA atoms found for superposition.")
                return float('nan')

            # Perform superposition
            super_imposer = Superimposer()
            super_imposer.set_atoms(ref_sup_atoms, target_sup_atoms)
            super_imposer.apply(target_structure.get_atoms())

            # Get atoms for RMSD calculation
            ref_rmsd_atoms = []
            target_rmsd_atoms = []
            for res_id in rmsd_indices:
                try:
                    ref_res = ref_structure[0][chain_id][res_id]
                    target_res = target_structure[0][chain_id][res_id]
                    ref_rmsd_atoms.append(ref_res["CA"])
                    target_rmsd_atoms.append(target_res["CA"])
                except KeyError:
                    logging.warning(f"Residue {res_id} not found for RMSD calculation. Skipping.")

            if not ref_rmsd_atoms or not target_rmsd_atoms:
                logging.warning("No CA atoms found for RMSD calculation.")
                return float('nan')

            # Calculate RMSD for specified residues
            rmsd = self._calculate_rmsd(ref_rmsd_atoms, target_rmsd_atoms)
            return rmsd

        except Exception as e:
            logging.error(f"Error calculating CA RMSD: {e}")
            raise

    def _calculate_rmsd(self, atoms1: List, atoms2: List) -> float:
        """Helper method to calculate RMSD between two lists of atoms."""
        if len(atoms1) != len(atoms2):
            raise ValueError("Atom lists must have same length")
        
        squared_sum = 0.0
        for a1, a2 in zip(atoms1, atoms2):
            diff = a1.get_coord() - a2.get_coord()
            squared_sum += np.dot(diff, diff)
        
        return np.sqrt(squared_sum / len(atoms1))

    def calculate_tm_score(
        self,
        target_pdb: str,
        reference_pdb: str,
        tm_align_path: str = DEFAULT_TMALIGN_PATH
    ) -> float:
        """
        Calculate TM-score between target and reference PDB structures.

        Args:
            target_pdb (str): Path to the target PDB file.
            reference_pdb (str): Path to the reference PDB file.
            tm_align_path (str, optional): Path to the TMalign executable. 
                Defaults to DEFAULT_TMALIGN_PATH.

        Returns:
            float: TM-score value.

        Raises:
            Exception: If an error occurs during TM-score calculation.
        """
        try:
            result = subprocess.run(
                [tm_align_path, target_pdb, reference_pdb],
                stdout=subprocess.PIPE,
                text=True,
                check=True
            )
            output = result.stdout
            
            tm_score = None
            for line in output.split('\n'):
                if "if normalized by length of Chain_2" in line:
                    tm_score = float(line.split()[1])
                    break
                    
            if tm_score is None:
                raise ValueError("Could not find TM-score in TMalign output")
                
            return tm_score
            
        except Exception as e:
            logging.error(f"Error calculating TM-score: {e}")
            raise

    def plddt_process(self,
                      pdb_file_path: str,
                      residue_indices: List[int]) -> Optional[float]:
        """
        Calculate the average pLDDT score for specified residues in a PDB file.

        Args:
            pdb_file_path (str): Path to the PDB file.
            residue_indices (List[int]): Residue indices to consider.

        Returns:
            Optional[float]: Average pLDDT score or None if not available.
        """
        try:
            b_factors = []
            with open(pdb_file_path, 'r') as pdb_file:
                for line in pdb_file:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        residue_index = int(line[22:26].strip())
                        if residue_index in residue_indices:
                            b_factor = float(line[60:66].strip())
                            b_factors.append(b_factor)

            if b_factors:
                average_b = sum(b_factors) / len(b_factors)
                return average_b
            else:
                return None
        except Exception as e:
            logging.error(f"Error processing pLDDT for {pdb_file_path}: {e}")
            return None

    def get_residue_center(
        self, structure: Any, chain_id: str, residue_indices: List[int]
    ) -> Optional[np.ndarray]:
        """
        Calculate the center of mass for a set of residues in a specific chain.

        Args:
            structure (Any): Parsed PDB structure.
            chain_id (str): Chain identifier.
            residue_indices (List[int]): Residue indices.

        Returns:
            Optional[np.ndarray]: Center of mass coordinates or None if residues not found.
        """
        try:
            atoms = []
            for res_index in residue_indices:
                try:
                    residue = structure[0][chain_id][res_index]
                    atoms.extend([atom for atom in residue.get_atoms()])
                except KeyError:
                    logging.error(f"Residue {res_index} not found in chain {chain_id}")
                    return None
            if not atoms:
                return None
            return self.calculate_com(atoms)
        except Exception as e:
            logging.error(f"Error calculating residue center: {e}")
            return None

    def get_atoms_from_residue_indices(
        self, structure: Any, residue_indices: List[int]
    ) -> List[Any]:
        """
        Retrieve all atoms from specified residue indices in a structure.

        Args:
            structure (Any): Parsed PDB structure.
            residue_indices (List[int]): Residue indices.

        Returns:
            List[Any]: List of atom objects.
        """
        atoms = []
        for residue in structure.get_residues():
            if residue.get_id()[1] in residue_indices:
                atoms.extend(residue.get_atoms())
        return atoms

def get_result_df(parent_dir: str, 
                  filter_criteria: List[Dict[str, Any]], 
                  basics: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate a DataFrame containing calculated properties for each PDB file in the directory.

    Args:
        parent_dir (str): Path to the parent directory containing PDB files.
        filter_criteria (List[Dict[str, Any]]): List of criteria for filtering and calculation.
        basics (Dict[str, Any]): Basic information required for calculations, usuallu full protein index list and rmsd refernce pdb.

    Returns:
        pd.DataFrame: DataFrame with results.
    """
    logging.info(f'Processing {parent_dir}')
    pdb_files = [
        os.path.join(dirpath, f)
        for dirpath, _, filenames in os.walk(parent_dir)
        for f in filenames
        if f.endswith('.pdb') and 'non_a3m' not in f
    ]
    logging.info(f'Found {len(pdb_files)} PDB files')

    properties_to_calculate = ['seq_count', 'plddt']
    for criterion in filter_criteria:
        if criterion['type'] not in properties_to_calculate:
            properties_to_calculate.append(criterion['type'])

    analyzer = StructureAnalyzer()

    def process_pdb(pdb: str) -> Dict[str, Any]:
        result = {'PDB': pdb}

        if 'seq_count' in properties_to_calculate:
            a3m_file = pdb.split("_unrelaxed")[0] + '.a3m'
            result['seq_count'] = count_sequences_in_a3m(a3m_file)
        if 'plddt' in properties_to_calculate:
            # Calculate full protein pLDDT if full_index is provided
            if 'full_index' in basics:
                full_index = []
                if isinstance(basics['full_index'], list):
                    # Multiple ranges
                    for range_dict in basics['full_index']:
                        full_index.extend(range(range_dict['start'], range_dict['end'] + 1))
                else:
                    # Single range (backward compatibility)
                    if isinstance(basics['full_index'], dict) and 'start' in basics['full_index'] and 'end' in basics['full_index']:
                        full_index = range(int(basics['full_index']['start']), int(basics['full_index']['end']) + 1)
                    elif isinstance(basics['full_index'], list):
                        full_index = [int(i) for i in basics['full_index']]
                result['plddt'] = analyzer.plddt_process(pdb, list(full_index))
            
            # Calculate local pLDDT if local_index is provided
            if 'local_index' in basics:
                local_index = []
                if isinstance(basics['local_index'], list):
                    # Multiple ranges
                    for range_dict in basics['local_index']:
                        local_index.extend(range(range_dict['start'], range_dict['end'] + 1))
                else:
                    # Single range (backward compatibility)
                    if isinstance(basics['local_index'], dict) and 'start' in basics['local_index'] and 'end' in basics['local_index']:
                        local_index = range(int(basics['local_index']['start']), int(basics['local_index']['end']) + 1)
                    elif isinstance(basics['local_index'], list):
                        local_index = [int(i) for i in basics['local_index']]
                result['local_plddt'] = analyzer.plddt_process(pdb, list(local_index))

        for criterion in filter_criteria:
            if criterion['type'] == 'distance':
                indices1 = criterion['indices']['set1']
                indices2 = criterion['indices']['set2']
                distance = analyzer.calculate_residue_distance(pdb, 'A', indices1, indices2)
                result[criterion['name']] = distance
            elif criterion['type'] == 'angle':
                angle = analyzer.calculate_angle(pdb, 
                                                 criterion['indices']['domain1'], 
                                                 criterion['indices']['domain2'], 
                                                 criterion['indices']['hinge'])
                result[criterion['name']] = angle
            elif criterion['type'] == 'rmsd':
                assert 'superposition_indices' in criterion and 'rmsd_indices' in criterion, "Both 'superposition_indices' and 'rmsd_indices' must be defined in the criterion"
                assert 'ref_pdb' in criterion, "'ref_pdb' must be defined in the criterion"
                
                # Handle superposition indices
                superposition_indices = []
                if isinstance(criterion['superposition_indices'], list):
                    # Multiple ranges
                    for range_dict in criterion['superposition_indices']:
                        superposition_indices.extend(range(range_dict['start'], range_dict['end'] + 1))
                else:
                    # Single range (backward compatibility)
                    superposition_indices = range(criterion['superposition_indices']['start'],
                                               criterion['superposition_indices']['end'] + 1)
                
                # Handle multiple ranges of RMSD indices
                rmsd_indices = []
                if isinstance(criterion['rmsd_indices'], list):
                    # Multiple ranges
                    for range_dict in criterion['rmsd_indices']:
                        rmsd_indices.extend(range(range_dict['start'], range_dict['end'] + 1))
                else:
                    # Single range (backward compatibility)
                    rmsd_indices = range(criterion['rmsd_indices']['start'],
                                      criterion['rmsd_indices']['end'] + 1)
                
                rmsd = analyzer.calculate_ca_rmsd(criterion['ref_pdb'], 
                                                pdb,
                                                list(superposition_indices),
                                                list(rmsd_indices),
                                                chain_id='A')
                result[criterion['name']] = rmsd
            elif criterion['type'] == 'tmscore':
                assert 'ref_pdb' in criterion, "'ref_pdb' must be defined in the criterion"
                tm_score = analyzer.calculate_tm_score(pdb, criterion['ref_pdb'])
                result[criterion['name']] = tm_score

        return result

    results = Parallel(n_jobs=-1)(delayed(process_pdb)(pdb) for pdb in tqdm(pdb_files, desc="Processing PDB files"))
    results_df = pd.DataFrame(results)

    return results_df



def get_protein_sequence(pdb_filename: str) -> str:
    """
    Extract the protein sequence from a PDB file.

    Args:
        pdb_filename (str): Path to the PDB file.

    Returns:
        str: The full protein sequence.

    Raises:
        Exception: If there's an error in processing the PDB file.
    """
    try:
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure("Protein", pdb_filename)
        ppb = PPBuilder()
        sequences = []
        for pp in ppb.build_peptides(structure):
            sequence = pp.get_sequence()
            sequences.append(str(sequence))

        full_sequence = ''.join(sequences)
        return full_sequence

    except Exception as e:
        logging.error(f"Error getting protein sequence: {e}")
        raise

def count_sequences_in_a3m(a3m_file: str) -> int:
    """
    Count the number of sequences in an A3M file.

    Args:
        a3m_file (str): Path to the A3M file.

    Returns:
        int: The number of sequences in the A3M file.
    """
    count = 0
    try:
        with open(a3m_file, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    count += 1
    except FileNotFoundError:
        logging.error(f"A3M file not found: {a3m_file}")
    return count

def load_filter_modes(file_path: str) -> Dict:
    """
    Load filter modes from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing filter modes.

    Returns:
        Dict: The loaded filter modes.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def apply_filters(df_threshold: pd.DataFrame, 
                  df_operate: pd.DataFrame, 
                  filter_criteria: List[Dict[str, Any]], 
                  quantile: float) -> pd.DataFrame:
    """
    Apply filters to the dataframe based on specified criteria and quantile thresholds.

    Args:
        df_threshold (pd.DataFrame): DataFrame used for calculating thresholds.
        df_operate (pd.DataFrame): DataFrame to apply filters on.
        filter_criteria (List[Dict[str, Any]]): List of filter criteria dictionaries.
        quantile (float): Quantile value for threshold calculation.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df_operate.copy()
    
    for criterion in filter_criteria:
        column_name = criterion['name']
        method = criterion['method']
        
        if method not in ['above', 'below']:
            logging.warning(f"Invalid filter method: {method}. Skipping this criterion.")
            continue
        
        if column_name not in df_threshold.columns:
            logging.warning(f"Column {column_name} not found in threshold DataFrame. Skipping this criterion.")
            continue
        
        if method == 'above':
            threshold_value = df_threshold[column_name].quantile(1 - quantile)
            filtered_df = filtered_df[filtered_df[column_name] > threshold_value]
            logging.info(f"Filtering {column_name} above {(1-quantile)*100}% quantile value: {threshold_value}")
        elif method == 'below':
            threshold_value = df_threshold[column_name].quantile(quantile)
            filtered_df = filtered_df[filtered_df[column_name] < threshold_value]
            logging.info(f"Filtering {column_name} below {quantile*100}% quantile value: {threshold_value}")
    
    return filtered_df
