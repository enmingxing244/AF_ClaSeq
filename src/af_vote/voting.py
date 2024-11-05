import os
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from af_vote.structure_analysis import StructureAnalyzer

@dataclass
class ProcessingResult:
    pdb_file: str
    properties: Dict[str, float]
    calculated_properties: Dict[str, float]

class VotingAnalyzer:
    def __init__(self, config: Dict[str, Any], num_bins: int = 10):
        self.config = config
        self.num_bins = num_bins
        self.structure_analyzer = StructureAnalyzer()
        self._setup_logging()

    @staticmethod
    def _setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_a3m_batch(self, batch_info: List[Tuple[str, str]], 
                         pdb_assignments: Dict[str, int], 
                         source_headers: Set[str]) -> Dict[str, List[int]]:
        """Process a batch of A3M files with improved error handling."""
        batch_votes = defaultdict(list)
        
        for root, a3m_file in batch_info:
            try:
                pdb_path = self._get_pdb_path(root, a3m_file)
                if pdb_path not in pdb_assignments:
                    continue

                assignment = pdb_assignments[pdb_path]
                self._process_a3m_file(root, a3m_file, assignment, source_headers, batch_votes)
                
            except Exception as e:
                logging.error(f"Error processing {a3m_file}: {e}", exc_info=True)
                
        return batch_votes

    def _get_pdb_path(self, root: str, a3m_file: str) -> str:
        """Generate PDB path from A3M file."""
        base_name = Path(a3m_file).stem
        pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
        return str(Path(root) / pdb_name)

    def _process_a3m_file(self, root: str, a3m_file: str, assignment: int, 
                         source_headers: Set[str], batch_votes: Dict[str, List[int]]) -> None:
        """Process individual A3M file."""
        a3m_path = Path(root) / a3m_file
        with open(a3m_path) as f:
            for line in f:
                if line.startswith('>'):
                    header = line.strip()[1:]
                    if header in source_headers:
                        batch_votes[header].append(assignment)

    def calculate_structure_properties(self, pdb_file: str) -> Dict[str, float]:
        """Calculate structural properties with improved parallelization."""
        properties = {}
        
        try:
            self._process_plddt_score(pdb_file, properties)
            self._process_criteria_parallel(pdb_file, properties)
            
        except Exception as e:
            logging.error(f"Error calculating properties for {pdb_file}: {e}", exc_info=True)
            
        return properties

    def _process_plddt_score(self, pdb_file: str, properties: Dict[str, float]) -> None:
        """Process pLDDT score calculation."""
        indices = self.config['indices']
        if 'full_index' in indices:
            full_index = indices['full_index']
            index_list = (list(range(full_index['start'], full_index['end'] + 1)) 
                        if isinstance(full_index, dict) else full_index)
            properties['plddt'] = self.structure_analyzer.plddt_process(pdb_file, index_list)

    def _process_criteria_parallel(self, pdb_file: str, properties: Dict[str, float]) -> None:
        """Process filter criteria in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor() as executor:
            futures = []
            for criterion in self.config['filter_criteria']:
                if criterion['type'] == 'distance':
                    futures.append(executor.submit(
                        self.structure_analyzer.calculate_residue_distance,
                        pdb_file,
                        'A',
                        criterion['indices']['set1'],
                        criterion['indices']['set2']
                    ))
                elif criterion['type'] == 'angle':
                    futures.append(executor.submit(
                        self.structure_analyzer.calculate_angle,
                        pdb_file,
                        criterion['indices']['domain1'],
                        criterion['indices']['domain2'],
                        criterion['indices']['hinge']
                    ))
                elif criterion['type'] == 'rmsd':
                    rmsd_indices = list(range(criterion['indices']['start'], 
                                           criterion['indices']['end'] + 1))
                    futures.append(executor.submit(
                        self.structure_analyzer.calculate_ca_rmsd,
                        self.config['indices']['reference_pdb'],
                        pdb_file,
                        rmsd_indices
                    ))
                elif criterion['type'] == 'tmscore':
                    futures.append(executor.submit(
                        self.structure_analyzer.calculate_tm_score,
                        pdb_file,
                        self.config['indices']['reference_pdb']
                    ))
                else:
                    logging.warning(f"Unknown criterion type: {criterion['type']}")
                    continue

            for future, criterion in zip(futures, self.config['filter_criteria']):
                try:
                    result = future.result()
                    if result is not None:
                        properties[criterion['name']] = result
                except Exception as e:
                    logging.error(f"Error processing criterion {criterion['name']}: {e}")

    def process_structures(self, pdb_files: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, Dict[str, float]]]:
        """Process structures with improved error handling and progress tracking."""
        all_properties = defaultdict(list)
        pdb_properties = {}
        calculated_properties = {}
        
        chunk_size = max(1, len(pdb_files) // (os.cpu_count() or 1))
        
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_structure_chunk, chunk, self.config)
                for chunk in np.array_split(pdb_files, chunk_size)
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing structures"):
                try:
                    chunk_results = future.result()
                    self._update_properties(chunk_results, all_properties, pdb_properties, calculated_properties)
                except Exception as e:
                    logging.error(f"Error processing structure chunk: {e}", exc_info=True)

        return self._create_bins_and_assignments(all_properties, pdb_properties)

    @staticmethod
    def _process_structure_chunk(chunk: List[str], config: Dict[str, Any]) -> List[ProcessingResult]:
        """Process a chunk of structures."""
        results = []
        for pdb_file in chunk:
            try:
                analyzer = VotingAnalyzer(config)
                properties = analyzer.calculate_structure_properties(pdb_file)
                
                if properties.get('plddt', 0) <= 75:
                    continue
                    
                calculated_props = {
                    criterion['name']: properties[criterion['name']]
                    for criterion in config['filter_criteria']
                    if criterion['name'] in properties
                }
                
                results.append(ProcessingResult(pdb_file, properties, calculated_props))
                
            except Exception as e:
                logging.error(f"Error processing {pdb_file}: {e}", exc_info=True)
                
        return results

    def _update_properties(self, results: List[ProcessingResult], 
                          all_properties: Dict[str, List[float]], 
                          pdb_properties: Dict[str, Dict[str, float]], 
                          calculated_properties: Dict[str, Dict[str, float]]) -> None:
        """Update property dictionaries with chunk results."""
        for result in results:
            for prop_name, value in result.properties.items():
                if prop_name != 'plddt':
                    all_properties[prop_name].append(value)
            pdb_properties[result.pdb_file] = result.properties
            calculated_properties[result.pdb_file] = result.calculated_properties

    def _create_bins_and_assignments(self, all_properties: Dict[str, List[float]], 
                                   pdb_properties: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, Dict[str, float]]]:
        """Create bins and assignments for properties."""
        bins = {
            prop_name: np.linspace(min(values), max(values), self.num_bins + 1)
            for prop_name, values in all_properties.items() if values
        }
        
        pdb_assignments = {}
        for pdb, properties in pdb_properties.items():
            for prop_name in bins:
                if prop_name in properties:
                    bin_idx = np.digitize(properties[prop_name], bins[prop_name]) - 1
                    pdb_assignments[pdb] = bin_idx
                    break
                    
        return bins, pdb_assignments, pdb_properties