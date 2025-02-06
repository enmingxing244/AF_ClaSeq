import os
import logging
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from af_vote.structure_analysis import StructureAnalyzer

class VotingAnalyzer:
    def __init__(self, max_workers: int = 32):
        self.max_workers = max_workers
        self._setup_logging()
        self.structure_analyzer = StructureAnalyzer()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_sampling_dirs(self, 
                              base_dir: str, 
                              filter_criteria: List[Dict[str, Any]], 
                              basics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Process all sampling directories to calculate metrics in parallel."""
        results = {}
        sampling_dirs = [d for d in os.listdir(base_dir) if d.startswith('sampling_')]
        
        # Collect all PDB files
        pdb_files = []
        for sampling_dir in sampling_dirs:
            dir_path = os.path.join(base_dir, sampling_dir)
            for f in os.listdir(dir_path):
                if f.endswith('.a3m'):
                    base_name = os.path.splitext(f)[0]
                    pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
                    pdb_path = os.path.join(dir_path, pdb_name)
                    if os.path.exists(pdb_path):
                        pdb_files.append((pdb_path, filter_criteria, basics))

        if not pdb_files:
            logging.warning("No PDB files found in sampling directories")
            return results

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_pdb_file, args) for args in pdb_files]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDB files"):
                result = future.result()
                if result:
                    pdb_path, metrics = result
                    results[pdb_path] = metrics
                    
        if not results:
            logging.warning("No valid results obtained from PDB processing")
            
        return results

    def _process_pdb_file(self, args):
        """Process a single PDB file and return metrics if valid."""
        pdb_path, filter_criteria, basics = args
        
        try:
            # Get pLDDT score using StructureAnalyzer
            plddt = self.structure_analyzer.plddt_process(pdb_path, list(range(basics['full_index']['start'], basics['full_index']['end'] + 1)))
            if not plddt:
                logging.warning(f"Could not calculate pLDDT for {pdb_path}")
                return None
                
            if plddt and plddt > 75:
                metrics = {}
                for criterion in filter_criteria:
                    if criterion['type'] == 'distance':
                        indices1 = criterion['indices']['set1']
                        indices2 = criterion['indices']['set2']
                        distance = self.structure_analyzer.calculate_residue_distance(pdb_path, 'A', indices1, indices2)
                        metrics[criterion['name']] = distance
                    elif criterion['type'] == 'angle':
                        angle = self.structure_analyzer.calculate_angle(pdb_path,
                                                         criterion['indices']['domain1'],
                                                         criterion['indices']['domain2'],
                                                         criterion['indices']['hinge'])
                        metrics[criterion['name']] = angle
                    elif criterion['type'] == 'rmsd':
                        superposition_indices = range(criterion['superposition_indices']['start'],
                                                   criterion['superposition_indices']['end'] + 1)
                        rmsd_indices = []
                        if isinstance(criterion['rmsd_indices'], list):
                            for range_dict in criterion['rmsd_indices']:
                                rmsd_indices.extend(range(range_dict['start'], range_dict['end'] + 1))
                        else:
                            rmsd_indices = range(criterion['rmsd_indices']['start'],
                                              criterion['rmsd_indices']['end'] + 1)
                        rmsd = self.structure_analyzer.calculate_ca_rmsd(basics['reference_pdb'],
                                                        pdb_path,
                                                        list(superposition_indices),
                                                        list(rmsd_indices),
                                                        chain_id='A')
                        if rmsd is not None:
                            metrics[criterion['name']] = rmsd
                        else:
                            logging.warning(f"Could not calculate RMSD for {pdb_path}")
                            return None
                    elif criterion['type'] == 'tmscore':
                        tm_score = self.structure_analyzer.calculate_tm_score(pdb_path, basics['reference_pdb'])
                        metrics[criterion['name']] = tm_score

                return pdb_path, metrics
        except Exception as e:
            logging.error(f"Error processing {pdb_path}: {str(e)}")
            return None
            
        return None

    def create_metric_bins(self, 
                           results: Dict[str, Dict[str, float]], 
                           metric_name: str, 
                           num_bins: int = 20) -> Tuple[np.ndarray, Dict[str, int]]:
        """Create bins for a specific metric and assign PDBs to bins."""
        if not results:
            raise ValueError("No results provided for binning")
            
        metric_values = [metrics[metric_name] for metrics in results.values() if metric_name in metrics]
        if not metric_values:
            raise ValueError(f"No valid values found for metric {metric_name}. Check if metric calculation succeeded.")
            
        bins = np.linspace(min(metric_values), max(metric_values), num_bins + 1)
        
        # Assign each PDB to a bin
        pdb_bins = {}
        for pdb, metrics in results.items():
            if metric_name in metrics:
                bin_idx = np.digitize(metrics[metric_name], bins) - 1
                pdb_bins[pdb] = bin_idx
            
        return bins, pdb_bins

    def get_sequence_votes(self, 
                           source_msa: str, 
                           sampling_base_dir: str, 
                           pdb_bins: Dict[str, int]) -> Dict[str, Tuple[int, int, int]]:
        """Get votes for each sequence based on metric bins."""
        if not os.path.exists(source_msa):
            raise FileNotFoundError(f"Source MSA file not found: {source_msa}")
            
        with open(source_msa) as f:
            source_headers = {line.strip()[1:] for line in f if line.startswith('>')}
            
        if not source_headers:
            raise ValueError("No headers found in source MSA file")
        
        # Collect all A3M files
        a3m_files = []
        for sampling_dir in os.listdir(sampling_base_dir):
            if sampling_dir.startswith('sampling_'):
                dir_path = os.path.join(sampling_base_dir, sampling_dir)
                for a3m_file in os.listdir(dir_path):
                    if a3m_file.endswith('.a3m'):
                        a3m_path = os.path.join(dir_path, a3m_file)
                        base_name = os.path.splitext(a3m_file)[0]
                        pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
                        pdb_file = os.path.join(dir_path, pdb_name)
                        if os.path.exists(pdb_file):
                            a3m_files.append((a3m_path, pdb_file))

        if not a3m_files:
            raise ValueError("No valid A3M/PDB file pairs found")

        batch_size = max(1, len(a3m_files) // (self.max_workers * 4))
        batches = [a3m_files[i:i + batch_size] for i in range(0, len(a3m_files), batch_size)]
        
        all_votes = defaultdict(list)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            batch_args = [(batch, pdb_bins, source_headers) for batch in batches]
            futures = [executor.submit(self._process_a3m_batch, args) for args in batch_args]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing A3M batches"):
                batch_votes = future.result()
                for header, votes in batch_votes.items():
                    all_votes[header].extend(votes)

        sequence_votes = {}
        for header, votes in all_votes.items():
            if votes:
                vote_array = np.array(votes)
                unique_votes, vote_counts = np.unique(vote_array, return_counts=True)
                most_common_idx = np.argmax(vote_counts)
                most_common_bin = unique_votes[most_common_idx]
                most_common_count = vote_counts[most_common_idx]
                total_votes = len(votes)
                
                sequence_votes[header] = (most_common_bin, most_common_count, total_votes)

        if not sequence_votes:
            logging.warning("No sequence votes were collected")

        return sequence_votes, dict(all_votes)

    def _process_a3m_batch(self, batch_args):
        """Process a batch of A3M files in parallel"""
        a3m_files_batch, pdb_bins, source_headers = batch_args
        batch_votes = defaultdict(list)
        
        for a3m_path, pdb_file in a3m_files_batch:
            if pdb_file in pdb_bins:
                try:
                    with open(a3m_path) as f:
                        headers = [line.strip()[1:] for line in f if line.startswith('>')]
                        
                    for header in headers:
                        if header in source_headers:
                            batch_votes[header].append(pdb_bins[pdb_file])
                except Exception as e:
                    logging.error(f"Error processing {a3m_path}: {str(e)}")
                        
        return dict(batch_votes)