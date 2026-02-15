#!/usr/bin/env python3
"""
Column Shuffle Pipeline for Comparative Conformational Analysis

This pipeline performs contact-based MSA column shuffling to assess
state-specific residue coupling in protein conformational transitions.

Usage:
    python column_shuffle_pipeline.py <config.yaml>

Example:
    python column_shuffle_pipeline.py configs/kaib_column_shuffle.yaml
"""

import sys
import json
import yaml
import random
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set, Optional
from Bio.PDB import PDBParser

# Import AF_ClaSeq utilities
from af_claseq.utils.logging_utils import setup_logger, get_logger
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.plotting_manager import create_2d_scatter_plot


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class GeneralConfig:
    """General pipeline configuration."""

    protein_name: str
    base_dir: str
    config_file: str  # Path to structure metrics JSON
    random_seed: int = 42
    use_composite_metrics: bool = False  # Enable composite metrics calculation

    # State 1 inputs
    state1_pdb: str = ""
    state1_a3m: str = ""
    state1_name: str = "state1"

    # State 2 inputs
    state2_pdb: str = ""
    state2_a3m: str = ""
    state2_name: str = "state2"

    # Column shuffle parameters
    num_shuffles: int = 20
    contact_threshold: float = 0.8
    sigmoid_center: float = 8.0
    sigmoid_steepness: float = 1.0
    min_sequence_separation: int = 10
    unique_pair_residue_range: Optional[List[int]] = None  # [start, end] to filter unique pairs, None = no filtering


@dataclass
class SlurmConfig:
    """SLURM job submission configuration."""

    conda_env_path: str
    slurm_account: str
    slurm_partition: str = "nextgen"
    slurm_time: str = "01:00:00"
    slurm_nodes: int = 1
    slurm_gpus_per_task: int = 1
    slurm_cpus_per_task: int = 8
    max_workers: int = 200
    num_models: int = 1
    num_seeds: int = 1  # Number of random seeds for ColabFold
    slurm_output: str = "/dev/null"
    slurm_error: str = "/dev/null"
    slurm_tasks: int = 1


@dataclass
class PlottingConfig:
    """Plotting configuration."""

    metric1_name: str  # e.g., "state1_tmscore"
    metric2_name: str  # e.g., "state2_tmscore"
    metric1_color: List[str]  # [start_color, end_color]
    metric2_color: List[str]

    # Axis limits
    metric1_min: float = 0.0
    metric1_max: float = 1.0
    metric2_min: float = 0.0
    metric2_max: float = 1.0

    # Axis ticks
    metric1_ticks: Optional[List[float]] = None
    metric2_ticks: Optional[List[float]] = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    general: GeneralConfig
    slurm: SlurmConfig
    plotting: PlottingConfig

    def validate_paths(self) -> None:
        """Validate that required input files exist."""
        errors = []

        # Check PDB files
        if not Path(self.general.state1_pdb).exists():
            errors.append(f"State1 PDB not found: {self.general.state1_pdb}")
        if not Path(self.general.state2_pdb).exists():
            errors.append(f"State2 PDB not found: {self.general.state2_pdb}")

        # Check MSA files
        if not Path(self.general.state1_a3m).exists():
            errors.append(f"State1 MSA not found: {self.general.state1_a3m}")
        if not Path(self.general.state2_a3m).exists():
            errors.append(f"State2 MSA not found: {self.general.state2_a3m}")

        # Check config file
        if not Path(self.general.config_file).exists():
            errors.append(f"Structure metrics config not found: {self.general.config_file}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))


def load_column_shuffle_config(yaml_path: str) -> PipelineConfig:
    """
    Load and parse YAML configuration file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        PipelineConfig instance with validated configuration

    Raises:
        ValueError: If required sections are missing or validation fails
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['general', 'slurm', 'plotting']
    missing_sections = [s for s in required_sections if s not in config_dict]
    if missing_sections:
        raise ValueError(f"Missing required sections in config: {', '.join(missing_sections)}")

    # Parse sections
    general_config = GeneralConfig(**config_dict['general'])
    slurm_config = SlurmConfig(**config_dict['slurm'])
    plotting_config = PlottingConfig(**config_dict['plotting'])

    # Create pipeline config
    pipeline_config = PipelineConfig(
        general=general_config,
        slurm=slurm_config,
        plotting=plotting_config
    )

    # Validate paths
    pipeline_config.validate_paths()

    return pipeline_config


# ============================================================================
# Pipeline Classes
# ============================================================================

class ContactMapBuilder:
    """
    Constructs contact probability maps from PDB files using CB atoms (CA for GLY).
    Computes difference maps and identifies unique residue pairs.
    """

    def __init__(self, sigmoid_center: float = 8.0, sigmoid_steepness: float = 1.0):
        """
        Initialize ContactMapBuilder.

        Args:
            sigmoid_center: Distance at which sigmoid = 0.5 (default: 8.0 Å)
            sigmoid_steepness: Controls sigmoid steepness (default: 1.0)
        """
        self.sigmoid_center = sigmoid_center
        self.sigmoid_steepness = sigmoid_steepness
        self.logger = get_logger(__name__)

    def extract_cb_coordinates(self, pdb_path: str) -> np.ndarray:
        """
        Extract CB coordinates (CA for glycine) from PDB file.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Array of shape (N_residues, 3) with coordinates

        Raises:
            ValueError: If PDB file cannot be parsed
        """
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("protein", pdb_path)
        except Exception as e:
            raise ValueError(f"Failed to parse PDB file {pdb_path}: {e}")

        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip hetero residues
                    if residue.id[0] != ' ':
                        continue

                    if residue.get_resname() == 'GLY':
                        # Use CA for glycine
                        if 'CA' in residue:
                            atom = residue['CA']
                            coords.append(atom.get_coord())
                    else:
                        # Use CB for all other residues
                        if 'CB' in residue:
                            atom = residue['CB']
                            coords.append(atom.get_coord())
                        elif 'CA' in residue:
                            # Fallback to CA if CB missing
                            atom = residue['CA']
                            coords.append(atom.get_coord())

        if not coords:
            raise ValueError(f"No valid coordinates extracted from {pdb_path}")

        return np.array(coords)

    def compute_pairwise_distances(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between all residues.

        Args:
            coords: Array of shape (N_residues, 3)

        Returns:
            Symmetric distance matrix of shape (N, N)
        """
        # Broadcasting: coords[:, None, :] - coords[None, :, :]
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2))
        return distances

    def sigmoid_normalization(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid normalization to distances to convert to contact probability.

        Formula: 1.0 / (1.0 + exp(steepness * (distance - center)))

        Args:
            distances: Distance matrix

        Returns:
            Contact probability matrix in range [0, 1]
        """
        return 1.0 / (1.0 + np.exp(self.sigmoid_steepness * (distances - self.sigmoid_center)))

    def build_contact_map(self, pdb_path: str) -> np.ndarray:
        """
        Complete pipeline: PDB -> coordinates -> distances -> contact map.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Contact probability map (N x N)
        """
        self.logger.info(f"Building contact map for {pdb_path}")
        coords = self.extract_cb_coordinates(pdb_path)
        self.logger.info(f"Extracted {len(coords)} residue coordinates")
        distances = self.compute_pairwise_distances(coords)
        contact_map = self.sigmoid_normalization(distances)
        return contact_map

    def compute_difference_map(self, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
        """
        Compute difference map: map1 - map2.

        Args:
            map1: First contact map
            map2: Second contact map

        Returns:
            Difference map (positive values indicate contacts unique to state1)
        """
        # Ensure same size
        min_size = min(map1.shape[0], map2.shape[0])
        diff_map = map1[:min_size, :min_size] - map2[:min_size, :min_size]
        return diff_map

    def identify_unique_pairs(
        self,
        diff_map: np.ndarray,
        threshold: float,
        min_sequence_separation: int,
        direction: str = 'positive',
        residue_range: Optional[List[int]] = None
    ) -> List[Tuple[int, int]]:
        """
        Identify unique residue pairs from difference map.

        Args:
            diff_map: Contact difference map
            threshold: Threshold for identifying unique pairs (e.g., 0.8)
            min_sequence_separation: Minimum sequence separation (e.g., 10)
            direction: 'positive' for state1 unique pairs (diff > threshold),
                      'negative' for state2 unique pairs (diff < -threshold)
            residue_range: Optional [start, end] range to filter pairs. Only pairs where
                          BOTH residues are within this range will be kept. None = no filtering.

        Returns:
            List of (residue_i, residue_j) tuples (1-indexed)
        """
        n = diff_map.shape[0]

        # Create sequence separation mask
        i_indices, j_indices = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        sep_mask = np.abs(i_indices - j_indices) >= min_sequence_separation

        # Apply threshold based on direction
        if direction == 'positive':
            threshold_mask = diff_map > threshold
        else:  # negative
            threshold_mask = diff_map < -threshold

        # Combine masks
        combined_mask = threshold_mask & sep_mask

        # Extract pairs
        i_coords, j_coords = np.where(combined_mask)
        pairs = [(int(i) + 1, int(j) + 1) for i, j in zip(i_coords, j_coords)]

        # Remove duplicates (i,j) and (j,i) by keeping only i < j
        unique_pairs = [(i, j) for i, j in pairs if i < j]

        self.logger.info(f"Identified {len(unique_pairs)} unique pairs before filtering ({direction})")

        # Filter by residue range if specified
        if residue_range is not None:
            start_res, end_res = residue_range
            filtered_pairs = [
                (i, j) for i, j in unique_pairs
                if start_res <= i <= end_res and start_res <= j <= end_res
            ]
            self.logger.info(f"Filtered to {len(filtered_pairs)} unique pairs within range [{start_res}, {end_res}]")
            unique_pairs = filtered_pairs

        self.logger.info(f"Final: {len(unique_pairs)} unique pairs ({direction})")
        return unique_pairs


class MSAColumnShuffler:
    """
    MSA column shuffling utility.
    Shuffles specified columns in MSA files while keeping the query sequence fixed.
    """

    def __init__(
        self,
        input_a3m: str,
        output_dir: str,
        num_shuffles: int,
        random_seed: int = None
    ):
        """
        Initialize MSA column shuffler.

        Args:
            input_a3m: Path to input MSA file
            output_dir: Output directory for shuffled MSAs
            num_shuffles: Number of shuffled replicates to generate
            random_seed: Random seed for reproducibility
        """
        self.input_a3m = Path(input_a3m)
        self.output_dir = Path(output_dir)
        self.num_shuffles = num_shuffles
        self.random_seed = random_seed
        self.logger = get_logger(__name__)

        # MSA data
        self.query_header = ""
        self.query_seq = ""
        self.headers = []
        self.sequences = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_msa(self) -> None:
        """
        Parse the input MSA file.
        The first sequence is always treated as the query sequence.
        Query header can be >query, >seq, >101, or any other format.
        """
        self.logger.info(f"Parsing MSA file: {self.input_a3m}")

        with open(self.input_a3m, 'r') as f:
            lines = f.readlines()

        current_seq = ""
        current_header = ""
        is_first_sequence = True

        for line in lines:
            line = line.strip()

            # Skip comment lines and empty lines
            if line.startswith('#') or not line:
                continue

            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header and current_seq:
                    if is_first_sequence:
                        # First sequence is always the query
                        self.query_header = current_header
                        self.query_seq = current_seq
                        is_first_sequence = False
                    else:
                        self.headers.append(current_header)
                        self.sequences.append(current_seq)

                # Start new sequence
                current_header = line
                current_seq = ""
            else:
                # Accumulate sequence
                current_seq += line

        # Add the last sequence
        if current_header and current_seq:
            if is_first_sequence:
                # First sequence is always the query
                self.query_header = current_header
                self.query_seq = current_seq
            else:
                self.headers.append(current_header)
                self.sequences.append(current_seq)

        self.logger.info(f"Parsed {len(self.sequences)} non-query sequences plus query sequence")
        self.logger.info(f"Query header: {self.query_header}")
        self.logger.info(f"Query sequence length: {len(self.query_seq)}")

    def shuffle_columns_at_positions(
        self,
        shuffle_positions: List[int]
    ) -> List[List[str]]:
        """
        Generate multiple shuffled versions of sequences.

        Args:
            shuffle_positions: List of column positions to shuffle (1-indexed)

        Returns:
            List of shuffled sequence sets
        """
        # Convert to 0-indexed
        zero_indexed_positions = [pos - 1 for pos in shuffle_positions]

        # Set random seed
        if self.random_seed is not None:
            random.seed(self.random_seed)

        all_shuffled_sequences = []

        for shuffle_idx in range(self.num_shuffles):
            # For each position, extract the column and shuffle it
            shuffled_columns = []
            for pos in zero_indexed_positions:
                # Extract column at this position from all sequences
                column = []
                for seq in self.sequences:
                    if pos < len(seq):
                        column.append(seq[pos])
                    else:
                        column.append('-')

                # Shuffle the column
                shuffled_column = column.copy()
                random.shuffle(shuffled_column)
                shuffled_columns.append(shuffled_column)

            # Create new sequences with shuffled residues
            new_sequences = []
            for seq_idx, original_seq in enumerate(self.sequences):
                # Start with original sequence as list
                new_seq_list = list(original_seq)

                # Replace residues at specified positions with shuffled ones
                for col_idx, pos in enumerate(zero_indexed_positions):
                    if pos < len(new_seq_list) and seq_idx < len(shuffled_columns[col_idx]):
                        new_seq_list[pos] = shuffled_columns[col_idx][seq_idx]

                new_sequences.append(''.join(new_seq_list))

            all_shuffled_sequences.append(new_sequences)

        return all_shuffled_sequences

    def save_shuffled_msa(
        self,
        shuffled_sequences: List[str],
        shuffle_index: int,
        state_name: str
    ) -> str:
        """
        Save shuffled MSA to file.

        Args:
            shuffled_sequences: List of shuffled sequences
            shuffle_index: Index of this shuffle (1-based)
            state_name: State name for filename

        Returns:
            Path to saved file
        """
        output_file = self.output_dir / f"{state_name}_shuffle_v{shuffle_index:02d}.a3m"

        # Write to file
        with open(output_file, 'w') as f:
            # Write query sequence first (unchanged)
            f.write(f"{self.query_header}\n")
            f.write(f"{self.query_seq}\n")

            # Write shuffled sequences
            for header, seq in zip(self.headers, shuffled_sequences):
                f.write(f"{header}\n")
                f.write(f"{seq}\n")

        return str(output_file)

    def generate_shuffled_msas(
        self,
        shuffle_positions: List[int],
        state_name: str
    ) -> List[str]:
        """
        Generate N shuffled MSA files.

        Args:
            shuffle_positions: List of column positions to shuffle (1-indexed)
            state_name: Name of the state (for output filenames)

        Returns:
            List of output file paths
        """
        self.logger.info(f"Generating {self.num_shuffles} shuffled MSAs for {state_name}")
        self.logger.info(f"Shuffling {len(shuffle_positions)} positions: {shuffle_positions}")

        # Parse MSA
        self.parse_msa()

        # Validate positions
        max_pos = max(shuffle_positions)
        if max_pos > len(self.query_seq):
            raise ValueError(f"Position {max_pos} exceeds sequence length {len(self.query_seq)}")

        # Generate shuffled versions
        all_shuffled_sequences = self.shuffle_columns_at_positions(shuffle_positions)

        # Save each shuffled version
        output_files = []
        for i, shuffled_sequences in enumerate(all_shuffled_sequences, 1):
            output_file = self.save_shuffled_msa(shuffled_sequences, i, state_name)
            output_files.append(output_file)

        self.logger.info(f"Generated {len(output_files)} shuffled MSA files")
        return output_files

    def extract_column_positions_from_pairs(
        self,
        unique_pairs: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Flatten unique pairs to get all involved column positions.

        Example: [(1,10), (1,15), (5,10)] -> [1, 5, 10, 15]

        Args:
            unique_pairs: List of (residue_i, residue_j) tuples

        Returns:
            Sorted list of unique column positions
        """
        positions = set()
        for i, j in unique_pairs:
            positions.add(i)
            positions.add(j)

        sorted_positions = sorted(list(positions))
        self.logger.info(f"Extracted {len(sorted_positions)} unique column positions from pairs")
        return sorted_positions


class ColumnShufflePipeline:
    """
    Main orchestrator for the column shuffle pipeline.

    Stages:
    1. Contact map construction
    2. Column shuffling
    3. Parallel structure prediction
    4. Structure analysis
    5. Plotting
    """

    def __init__(self, yaml_config_path: str):
        """
        Initialize pipeline with YAML configuration.

        Args:
            yaml_config_path: Path to YAML configuration file
        """
        self.config = load_column_shuffle_config(yaml_config_path)
        self.logger = self._setup_logging()
        self._create_directories()
        self.slurm_submitter = None  # Initialized when needed

        self.logger.info("=" * 80)
        self.logger.info("COLUMN SHUFFLE PIPELINE INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"Protein: {self.config.general.protein_name}")
        self.logger.info(f"Base directory: {self.config.general.base_dir}")
        self.logger.info(f"Random seed: {self.config.general.random_seed}")
        self.logger.info("=" * 80)

    def _setup_logging(self) -> logging.Logger:
        """Set up centralized logging."""
        base_dir = Path(self.config.general.base_dir)
        log_dir = base_dir / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / "column_shuffle_pipeline.log"

        return setup_logger(
            name="af_claseq",
            log_file=str(log_file),
            level=logging.INFO,
            propagate=False,
            add_console_handler=True
        )

    def _create_directories(self) -> None:
        """Create necessary output directories."""
        base_dir = Path(self.config.general.base_dir)
        base_dir.mkdir(exist_ok=True, parents=True)

        # Create stage directories
        stages = [
            "01_contact_maps",
            "02_shuffled_msas",
            "03_predictions",
            "04_analysis",
            "05_plots"
        ]

        for stage in stages:
            (base_dir / stage).mkdir(exist_ok=True)

    def _init_slurm_submitter(self) -> SlurmJobSubmitter:
        """Initialize SLURM job submitter."""
        if self.slurm_submitter is None:
            self.slurm_submitter = SlurmJobSubmitter(
                conda_env_path=self.config.slurm.conda_env_path,
                slurm_account=self.config.slurm.slurm_account,
                slurm_partition=self.config.slurm.slurm_partition,
                slurm_time=self.config.slurm.slurm_time,
                slurm_output=self.config.slurm.slurm_output,
                slurm_error=self.config.slurm.slurm_error,
                slurm_nodes=self.config.slurm.slurm_nodes,
                slurm_gpus_per_task=self.config.slurm.slurm_gpus_per_task,
                slurm_cpus_per_task=self.config.slurm.slurm_cpus_per_task,
                max_workers=self.config.slurm.max_workers,
                num_models=self.config.slurm.num_models,
                num_seeds=self.config.slurm.num_seeds,
                random_seed=self.config.general.random_seed
            )
        return self.slurm_submitter

    def _plot_contact_maps(
        self,
        contact_map1: np.ndarray,
        contact_map2: np.ndarray,
        diff_map: np.ndarray,
        state1_name: str,
        state2_name: str,
        output_dir: Path
    ) -> None:
        """
        Generate and save contact map visualizations.

        Args:
            contact_map1: Contact map for state 1
            contact_map2: Contact map for state 2
            diff_map: Difference map (map1 - map2)
            state1_name: Name of state 1
            state2_name: Name of state 2
            output_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt

        self.logger.info("Creating contact map visualizations")

        # Plot state1 contact map
        plt.figure(figsize=(10, 8))
        plt.imshow(contact_map1, cmap='viridis', vmin=0, vmax=1, origin='lower')
        plt.colorbar(label='Contact Probability', shrink=0.8)
        plt.title(f'{state1_name} Contact Map', fontsize=14, fontweight='bold')
        plt.xlabel('Residue Index', fontsize=12)
        plt.ylabel('Residue Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f'{state1_name}_contact_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved {state1_name}_contact_map.png")

        # Plot state2 contact map
        plt.figure(figsize=(10, 8))
        plt.imshow(contact_map2, cmap='viridis', vmin=0, vmax=1, origin='lower')
        plt.colorbar(label='Contact Probability', shrink=0.8)
        plt.title(f'{state2_name} Contact Map', fontsize=14, fontweight='bold')
        plt.xlabel('Residue Index', fontsize=12)
        plt.ylabel('Residue Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f'{state2_name}_contact_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved {state2_name}_contact_map.png")

        # Plot difference map
        plt.figure(figsize=(10, 8))
        plt.imshow(diff_map, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        plt.colorbar(label='Contact Difference (State1 - State2)', shrink=0.8)
        plt.title('Contact Difference Map', fontsize=14, fontweight='bold')
        plt.xlabel('Residue Index', fontsize=12)
        plt.ylabel('Residue Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'difference_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved difference_map.png")

    def run_contact_map_construction(self) -> bool:
        """
        Stage 1: Construct contact maps and identify unique pairs.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: CONTACT MAP CONSTRUCTION")
        self.logger.info("=" * 80)

        try:
            # Set random seed
            np.random.seed(self.config.general.random_seed)

            # Initialize contact map builder
            builder = ContactMapBuilder(
                sigmoid_center=self.config.general.sigmoid_center,
                sigmoid_steepness=self.config.general.sigmoid_steepness
            )

            # Build contact maps for both states
            self.logger.info(f"Building contact map for state1: {self.config.general.state1_pdb}")
            contact_map1 = builder.build_contact_map(self.config.general.state1_pdb)

            self.logger.info(f"Building contact map for state2: {self.config.general.state2_pdb}")
            contact_map2 = builder.build_contact_map(self.config.general.state2_pdb)

            # Validate sequence lengths match
            if contact_map1.shape[0] != contact_map2.shape[0]:
                self.logger.warning(
                    f"Sequence length mismatch: state1={contact_map1.shape[0]}, "
                    f"state2={contact_map2.shape[0]}. Using minimum length."
                )

            # Compute difference map
            self.logger.info("Computing difference map")
            diff_map = builder.compute_difference_map(contact_map1, contact_map2)

            # Identify unique pairs for each state
            self.logger.info(f"Identifying unique pairs (threshold={self.config.general.contact_threshold})")
            if self.config.general.unique_pair_residue_range:
                self.logger.info(f"Residue range filter: {self.config.general.unique_pair_residue_range}")

            unique_pairs_state1 = builder.identify_unique_pairs(
                diff_map=diff_map,
                threshold=self.config.general.contact_threshold,
                min_sequence_separation=self.config.general.min_sequence_separation,
                direction='positive',
                residue_range=self.config.general.unique_pair_residue_range
            )

            unique_pairs_state2 = builder.identify_unique_pairs(
                diff_map=diff_map,
                threshold=self.config.general.contact_threshold,
                min_sequence_separation=self.config.general.min_sequence_separation,
                direction='negative',
                residue_range=self.config.general.unique_pair_residue_range
            )

            # Save outputs
            output_dir = Path(self.config.general.base_dir) / "01_contact_maps"

            np.save(output_dir / "state1_contact_map.npy", contact_map1)
            np.save(output_dir / "state2_contact_map.npy", contact_map2)
            np.save(output_dir / "difference_map.npy", diff_map)

            unique_pairs_dict = {
                self.config.general.state1_name: unique_pairs_state1,
                self.config.general.state2_name: unique_pairs_state2
            }

            with open(output_dir / "unique_pairs.json", 'w') as f:
                json.dump(unique_pairs_dict, f, indent=2)

            # Generate contact map visualizations
            self._plot_contact_maps(
                contact_map1=contact_map1,
                contact_map2=contact_map2,
                diff_map=diff_map,
                state1_name=self.config.general.state1_name,
                state2_name=self.config.general.state2_name,
                output_dir=output_dir
            )

            self.logger.info("Contact map construction completed successfully")
            self.logger.info(f"State1 unique pairs: {len(unique_pairs_state1)}")
            self.logger.info(f"State2 unique pairs: {len(unique_pairs_state2)}")

            return True

        except Exception as e:
            self.logger.error(f"Contact map construction failed: {e}", exc_info=True)
            return False

    def run_column_shuffling(self) -> bool:
        """
        Stage 2: Shuffle MSA columns at unique pair positions.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: COLUMN SHUFFLING")
        self.logger.info("=" * 80)

        try:
            # Set random seed
            random.seed(self.config.general.random_seed)
            np.random.seed(self.config.general.random_seed)

            # Load unique pairs
            unique_pairs_path = Path(self.config.general.base_dir) / "01_contact_maps" / "unique_pairs.json"
            with open(unique_pairs_path, 'r') as f:
                unique_pairs_dict = json.load(f)

            # Process each state
            for state_name, state_pdb, state_a3m in [
                (self.config.general.state1_name, self.config.general.state1_pdb, self.config.general.state1_a3m),
                (self.config.general.state2_name, self.config.general.state2_pdb, self.config.general.state2_a3m)
            ]:
                self.logger.info(f"Processing {state_name}")

                # Get unique pairs for this state
                unique_pairs = unique_pairs_dict[state_name]

                if not unique_pairs:
                    self.logger.warning(f"No unique pairs found for {state_name}, skipping")
                    continue

                # Create output directory
                output_dir = Path(self.config.general.base_dir) / "02_shuffled_msas" / state_name

                # Initialize shuffler
                shuffler = MSAColumnShuffler(
                    input_a3m=state_a3m,
                    output_dir=str(output_dir),
                    num_shuffles=self.config.general.num_shuffles,
                    random_seed=self.config.general.random_seed
                )

                # Extract column positions
                column_positions = shuffler.extract_column_positions_from_pairs(unique_pairs)

                # Generate shuffled MSAs
                output_files = shuffler.generate_shuffled_msas(
                    shuffle_positions=column_positions,
                    state_name=state_name
                )

                self.logger.info(f"Generated {len(output_files)} shuffled MSAs for {state_name}")

            self.logger.info("Column shuffling completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Column shuffling failed: {e}", exc_info=True)
            return False

    def run_parallel_prediction(self) -> bool:
        """
        Stage 3: Submit parallel ColabFold jobs for all shuffled MSAs.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: PARALLEL STRUCTURE PREDICTION")
        self.logger.info("=" * 80)

        try:
            # Initialize SLURM submitter
            submitter = self._init_slurm_submitter()

            # Collect all job specifications
            all_job_folders = []
            all_job_ids = []

            base_dir = Path(self.config.general.base_dir)
            shuffled_dir = base_dir / "02_shuffled_msas"
            predictions_dir = base_dir / "03_predictions"

            for state_name in [self.config.general.state1_name, self.config.general.state2_name]:
                state_shuffled_dir = shuffled_dir / state_name
                state_predictions_dir = predictions_dir / state_name

                if not state_shuffled_dir.exists():
                    self.logger.warning(f"Shuffled MSAs directory not found for {state_name}, skipping")
                    continue

                # Find all shuffled MSA files
                shuffled_files = sorted(state_shuffled_dir.glob(f"{state_name}_shuffle_v*.a3m"))

                for shuffled_file in shuffled_files:
                    # Extract shuffle index from filename
                    shuffle_idx = shuffled_file.stem.split('_v')[-1]

                    # Create prediction directory
                    pred_dir = state_predictions_dir / f"shuffle_v{shuffle_idx}"
                    pred_dir.mkdir(parents=True, exist_ok=True)

                    # Copy MSA file to prediction directory
                    shutil.copy(shuffled_file, pred_dir / shuffled_file.name)

                    # Add to job list
                    all_job_folders.append(str(pred_dir))
                    job_id = f"{self.config.general.protein_name}_{state_name}_v{shuffle_idx}"
                    all_job_ids.append(job_id)

            if not all_job_folders:
                self.logger.error("No shuffled MSAs found to process")
                return False

            self.logger.info(f"Submitting {len(all_job_folders)} prediction jobs")

            # Submit all jobs in parallel
            submitter.process_folders_concurrently(
                folders=all_job_folders,
                job_ids=all_job_ids,
                max_workers=self.config.slurm.max_workers
            )

            self.logger.info("All prediction jobs completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Parallel prediction failed: {e}", exc_info=True)
            return False

    def run_structure_analysis(self) -> bool:
        """
        Stage 4: Analyze predicted structures using user-defined metrics.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 4: STRUCTURE ANALYSIS")
        self.logger.info("=" * 80)

        try:
            # Load structure metrics config
            with open(self.config.general.config_file, 'r') as f:
                metrics_config = json.load(f)

            self.logger.info(f"Loaded metrics config from {self.config.general.config_file}")

            # Initialize structure analyzer
            analyzer = StructureAnalyzer()
            import pandas as pd

            base_dir = Path(self.config.general.base_dir)
            predictions_dir = base_dir / "03_predictions"

            # Analyze EACH state separately and save to separate files
            for state_name in [self.config.general.state1_name, self.config.general.state2_name]:
                self.logger.info(f"Analyzing structures for {state_name}")

                state_predictions_dir = predictions_dir / state_name

                if not state_predictions_dir.exists():
                    self.logger.warning(f"Predictions directory not found for {state_name}, skipping")
                    continue

                # Run analysis for this state only
                # Pass composite_metrics only if use_composite_metrics is enabled
                composite_metrics = None
                if self.config.general.use_composite_metrics:
                    composite_metrics = metrics_config.get('composite_metrics', [])

                results_df = analyzer.get_result_df(
                    parent_dir=str(state_predictions_dir),
                    filter_criteria=metrics_config.get('filter_criteria', []),
                    basics=metrics_config.get('basics', {}),
                    plddt_threshold=0,
                    composite_metrics=composite_metrics
                )

                # Add metadata columns
                results_df['state'] = state_name
                results_df['protein'] = self.config.general.protein_name

                # Save to STATE-SPECIFIC directory
                analysis_dir = base_dir / "04_analysis" / state_name
                analysis_dir.mkdir(parents=True, exist_ok=True)

                output_file = analysis_dir / "structure_analysis.csv"
                results_df.to_csv(output_file, index=False)

                self.logger.info(f"{state_name}: {len(results_df)} structures analyzed")
                self.logger.info(f"Results saved to {output_file}")

            return True

        except Exception as e:
            self.logger.error(f"Structure analysis failed: {e}", exc_info=True)
            return False

    def run_plotting(self) -> bool:
        """
        Stage 5: Generate publication-quality plots.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 5: PLOTTING")
        self.logger.info("=" * 80)

        try:
            import pandas as pd

            base_dir = Path(self.config.general.base_dir)

            # Generate plots for EACH state separately
            for state_name in [self.config.general.state1_name, self.config.general.state2_name]:
                self.logger.info(f"Generating plots for {state_name}")

                # Load state-specific analysis results
                analysis_dir = base_dir / "04_analysis" / state_name
                results_file = analysis_dir / "structure_analysis.csv"

                if not results_file.exists():
                    self.logger.warning(f"Analysis file not found for {state_name}, skipping")
                    continue

                results_df = pd.read_csv(results_file)
                self.logger.info(f"Loaded {len(results_df)} results for {state_name}")

                # Create STATE-SPECIFIC plots directory
                plots_dir = base_dir / "05_plots" / state_name
                plots_dir.mkdir(parents=True, exist_ok=True)

                # Generate 2D scatter plot for this state with pLDDT colormap
                create_2d_scatter_plot(
                    results_df=results_df,
                    metric_name1=self.config.plotting.metric1_name,
                    metric_name2=self.config.plotting.metric2_name,
                    output_dir=str(plots_dir),
                    color_metric='plddt',  # Color by pLDDT
                    cmap_colors=None,  # Use default pLDDT colormap
                    x_min=self.config.plotting.metric1_min,
                    x_max=self.config.plotting.metric1_max,
                    y_min=self.config.plotting.metric2_min,
                    y_max=self.config.plotting.metric2_max,
                    x_ticks=self.config.plotting.metric1_ticks,
                    y_ticks=self.config.plotting.metric2_ticks,
                    title=f"{self.config.general.protein_name} - {state_name} Column Shuffle"
                )

                self.logger.info(f"Plots saved to {plots_dir}")

            self.logger.info("Plotting completed for all states")
            return True

        except Exception as e:
            self.logger.error(f"Plotting failed: {e}", exc_info=True)
            return False

    def run_show_positions(self) -> bool:
        """
        Stage 6: Show aligned residues at unique pair positions from MSA.

        Extracts and displays the alignment at positions involved in unique pairs
        for each state, saved to a text file for easy visualization.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 6: SHOW UNIQUE PAIR POSITIONS IN MSA")
        self.logger.info("=" * 80)

        try:
            base_dir = Path(self.config.general.base_dir)

            # Load unique pairs
            unique_pairs_path = base_dir / "01_contact_maps" / "unique_pairs.json"
            with open(unique_pairs_path, 'r') as f:
                unique_pairs_dict = json.load(f)

            # Process each state
            for state_name, state_a3m in [
                (self.config.general.state1_name, self.config.general.state1_a3m),
                (self.config.general.state2_name, self.config.general.state2_a3m)
            ]:
                self.logger.info(f"Extracting positions for {state_name}")

                # Get unique pairs for this state
                unique_pairs = unique_pairs_dict.get(state_name, [])

                if not unique_pairs:
                    self.logger.warning(f"No unique pairs found for {state_name}, skipping")
                    continue

                # Flatten pairs to get unique positions (sorted)
                positions = sorted(set(pos for pair in unique_pairs for pos in pair))

                self.logger.info(f"Extracting {len(positions)} positions: {positions}")

                # Read MSA file
                with open(state_a3m, 'r') as f:
                    msa_lines = f.readlines()

                # Parse MSA
                sequences = []
                headers = []
                current_header = ""
                current_seq = ""

                for line in msa_lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    if line.startswith('>'):
                        if current_header and current_seq:
                            headers.append(current_header)
                            sequences.append(current_seq)
                        current_header = line
                        current_seq = ""
                    else:
                        current_seq += line

                # Add last sequence
                if current_header and current_seq:
                    headers.append(current_header)
                    sequences.append(current_seq)

                # Create output directory
                output_dir = base_dir / "06_position_alignments"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Extract columns at specified positions
                output_file = output_dir / f"{state_name}_unique_positions.txt"

                with open(output_file, 'w') as out:
                    # Write header
                    out.write(f"# Aligned residues at unique pair positions for {state_name}\n")
                    out.write(f"# Unique pairs: {unique_pairs}\n")
                    out.write(f"# Positions (1-indexed): {positions}\n")
                    out.write(f"# Total sequences: {len(sequences)}\n")
                    out.write("=" * 100 + "\n\n")

                    # ===== SECTION 1: All positions in order =====
                    out.write("SECTION 1: All Unique Positions\n")
                    out.write("-" * 100 + "\n")
                    out.write("Position:  ")
                    for pos in positions:
                        out.write(f"{pos:4d} ")
                    out.write("\n")
                    out.write("-" * 100 + "\n")

                    # Write each sequence
                    for idx, (header, seq) in enumerate(zip(headers, sequences)):
                        # Extract residues at specified positions (convert to 0-indexed)
                        residues = []
                        for pos in positions:
                            pos_idx = pos - 1  # Convert to 0-indexed
                            if pos_idx < len(seq):
                                residues.append(seq[pos_idx])
                            else:
                                residues.append('-')

                        # Better header truncation - show beginning and end with ...
                        if len(header) > 20:
                            # Keep more of the unique part (last characters)
                            display_header = header[:12] + "..." + header[-5:]
                        else:
                            display_header = header

                        # Add tab separator between header and residues
                        out.write(f"{display_header:<20}\t")

                        for res in residues:
                            out.write(f"{res:4s} ")
                        out.write("\n")

                    out.write("\n\n")

                    # ===== SECTION 2: Paired view =====
                    out.write("SECTION 2: Residue Pairs (Contact Pairs)\n")
                    out.write("=" * 100 + "\n")

                    # Show each pair
                    for pair_idx, (pos1, pos2) in enumerate(unique_pairs):
                        out.write(f"\nPair {pair_idx + 1}: [{pos1}, {pos2}]\n")
                        out.write("-" * 50 + "\n")
                        out.write(f"{'Sequence':<20}\tPos {pos1:3d}\tPos {pos2:3d}\n")
                        out.write("-" * 50 + "\n")

                        # Extract residues for this pair
                        for header, seq in zip(headers, sequences):
                            # Get residues at both positions
                            idx1, idx2 = pos1 - 1, pos2 - 1
                            res1 = seq[idx1] if idx1 < len(seq) else '-'
                            res2 = seq[idx2] if idx2 < len(seq) else '-'

                            # Truncate header
                            if len(header) > 20:
                                display_header = header[:12] + "..." + header[-5:]
                            else:
                                display_header = header

                            out.write(f"{display_header:<20}\t{res1:^3s}\t{res2:^3s}\n")

                    out.write("\n")

                self.logger.info(f"Position alignment saved to {output_file}")
                self.logger.info(f"Extracted {len(sequences)} sequences at {len(positions)} positions")

            self.logger.info("Position extraction completed for all states")
            return True

        except Exception as e:
            self.logger.error(f"Position extraction failed: {e}", exc_info=True)
            return False

    def run(self, stages: List[str] = None) -> None:
        """
        Run the complete pipeline or specific stages.

        Args:
            stages: List of stage names to run. If None, run all stages.
                   Valid stages: contact_maps, shuffle, predict, analyze, plot, show_positions
        """
        # Define all stages
        all_stages = {
            'contact_maps': self.run_contact_map_construction,
            'shuffle': self.run_column_shuffling,
            'predict': self.run_parallel_prediction,
            'analyze': self.run_structure_analysis,
            'plot': self.run_plotting,
            'show_positions': self.run_show_positions
        }

        # Determine which stages to run
        if stages is None:
            stages_to_run = list(all_stages.keys())
        else:
            stages_to_run = stages

        # Run each stage
        for stage_name in stages_to_run:
            if stage_name not in all_stages:
                self.logger.error(f"Unknown stage: {stage_name}")
                continue

            stage_func = all_stages[stage_name]
            success = stage_func()

            if not success:
                self.logger.error(f"Pipeline failed at stage: {stage_name}")
                return

        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)

    def print_execution_plan(self) -> None:
        """Print execution plan for dry-run mode."""
        print("=" * 80)
        print("COLUMN SHUFFLE PIPELINE - EXECUTION PLAN")
        print("=" * 80)
        print(f"\nProtein: {self.config.general.protein_name}")
        print(f"Base Directory: {self.config.general.base_dir}")
        print(f"\nState 1: {self.config.general.state1_name}")
        print(f"  PDB: {self.config.general.state1_pdb}")
        print(f"  MSA: {self.config.general.state1_a3m}")
        print(f"\nState 2: {self.config.general.state2_name}")
        print(f"  PDB: {self.config.general.state2_pdb}")
        print(f"  MSA: {self.config.general.state2_a3m}")
        print(f"\nParameters:")
        print(f"  Number of shuffles: {self.config.general.num_shuffles}")
        print(f"  Contact threshold: {self.config.general.contact_threshold}")
        print(f"  Sigmoid center: {self.config.general.sigmoid_center}")
        print(f"  Sigmoid steepness: {self.config.general.sigmoid_steepness}")
        print(f"  Min sequence separation: {self.config.general.min_sequence_separation}")
        print(f"\nStages:")
        print("  1. Contact map construction")
        print("  2. Column shuffling")
        print("  3. Parallel structure prediction")
        print("  4. Structure analysis")
        print("  5. Plotting")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Column Shuffle Pipeline for Conformational Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.yaml
  %(prog)s configs/kaib_example.yaml --stages contact_maps shuffle
  %(prog)s config.yaml --dry-run

Configuration:
  The YAML config file must specify:
  - general: protein name, base directory, input PDB/MSA files
  - slurm: job submission parameters
  - plotting: visualization parameters

For more information, see the plan file.
        """
    )

    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config and show execution plan without running'
    )

    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['contact_maps', 'shuffle', 'predict', 'analyze', 'plot', 'show_positions'],
        default=None,
        help='Run only specific stages (default: all stages)'
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        # Initialize pipeline
        pipeline = ColumnShufflePipeline(str(config_path))

        if args.dry_run:
            pipeline.print_execution_plan()
            sys.exit(0)

        # Run pipeline
        pipeline.run(stages=args.stages)

    except Exception as e:
        print(f"ERROR: Pipeline initialization or execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
