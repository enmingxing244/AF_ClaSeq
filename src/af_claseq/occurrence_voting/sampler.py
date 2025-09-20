"""
Random sampling module for occurrence voting.

This module handles random sampling of sequences from A3M files
without replacement, creating multiple small groups for structure prediction.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Use existing AF_ClaSeq utilities
from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m
from af_claseq.utils.logging_utils import get_logger


class SequenceSampler:
    """
    Handles random sampling of sequences from A3M files.

    This class implements sampling without replacement to create
    multiple groups for occurrence voting analysis.
    """

    def __init__(self, config, logger=None):
        """
        Initialize the sequence sampler.

        Args:
            config: OccurrenceVotingConfig object
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or get_logger("sequence_sampler")

        # Set random seed for reproducibility
        random.seed(config.general.random_seed)

        # Extract homodimer mode from structure prediction config
        self.homodimer_mode = config.structure_prediction.prediction_mode == "homodimer"

        # Note: No longer creating individual group directories
        # Groups are created directly in batch directories

        self.logger.info(f"Initialized SequenceSampler with seed: {config.general.random_seed}")
        self.logger.info(f"Prediction mode: {config.structure_prediction.prediction_mode}")

    def create_random_groups(self) -> List[Dict[str, any]]:
        """
        Create multiple random groups from source A3M file.

        Returns:
            List of group information dictionaries
        """
        self.logger.info(f"Reading sequences from: {self.config.general.source_a3m}")

        # Read sequences using existing utility
        sequences_dict = read_a3m_to_dict(self.config.general.source_a3m)
        sequence_items = list(sequences_dict.items())

        if len(sequence_items) < 2:
            raise ValueError(f"Need at least 2 sequences (query + others), found: {len(sequence_items)}")

        # Extract query sequence and non-query sequences
        query_header, query_sequence = sequence_items[0]
        available_sequences = sequence_items[1:]  # Skip query

        self.logger.info(f"Found {len(available_sequences)} non-query sequences for sampling")

        # Check if we have enough sequences
        min_needed = self.config.sampling.group_size
        if len(available_sequences) < min_needed:
            raise ValueError(f"Need at least {min_needed} non-query sequences, found: {len(available_sequences)}")

        # Perform random sampling ensuring no duplicates within each group
        groups_info = []
        num_groups = self.config.sampling.num_groups
        group_size = self.config.sampling.group_size

        # Check if we have enough unique sequences for each group
        if len(available_sequences) < group_size:
            raise ValueError(
                f"Cannot create groups of size {group_size} with only {len(available_sequences)} "
                f"unique sequences available. Each group requires {group_size} unique sequences."
            )

        # Create batches directly instead of individual groups
        batches_info = self._create_batches_directly(
            num_groups, group_size, available_sequences, query_header, query_sequence
        )

        self.logger.info(f"Created {len(batches_info)} batches containing {num_groups} groups total")
        return batches_info

    def _create_batches_directly(self, num_groups: int, group_size: int,
                               available_sequences: List[Tuple[str, str]],
                               query_header: str, query_sequence: str) -> List[Dict[str, Any]]:
        """
        Create batches directly without intermediate group files.

        Args:
            num_groups: Total number of groups to create
            group_size: Number of sequences per group
            available_sequences: Available sequences for sampling
            query_header: Query sequence header
            query_sequence: Query sequence

        Returns:
            List of batch information dictionaries
        """
        import math

        num_batches = self.config.sampling.num_batches
        groups_per_batch = math.ceil(num_groups / num_batches)

        batches_info = []
        group_counter = 0

        for batch_idx in range(num_batches):
            batch_id = f"batch_{batch_idx+1:03d}"
            batch_dir = self.config.get_batches_dir() / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)

            # Determine how many groups for this batch
            groups_in_this_batch = min(groups_per_batch, num_groups - group_counter)
            batch_groups = []

            # Create groups for this batch
            for group_in_batch in range(groups_in_this_batch):
                group_idx = group_counter + group_in_batch
                group_id = f"group_{group_idx+1:06d}"

                # Sample sequences for this group
                group_sequences = random.sample(available_sequences, group_size)

                # Create A3M file directly in batch directory
                a3m_filename = f"{group_id}.a3m"
                a3m_filepath = batch_dir / a3m_filename

                # Create sequences dictionary (query + group sequences)
                sequences_dict = {query_header: query_sequence}
                for header, sequence in group_sequences:
                    sequences_dict[header] = sequence

                # Write A3M file
                write_a3m(sequences_dict, a3m_filepath, homodimer_mode=self.homodimer_mode)

                batch_groups.append({
                    'group_id': group_id,
                    'group_index': group_idx,
                    'a3m_file': str(a3m_filepath),
                    'sequences_count': len(sequences_dict)
                })

            # Update group counter
            group_counter += groups_in_this_batch

            # Create batch info
            batch_info = {
                'batch_id': batch_id,
                'batch_index': batch_idx,
                'batch_dir': str(batch_dir),
                'groups': batch_groups,
                'groups_count': len(batch_groups)
            }

            batches_info.append(batch_info)

            self.logger.info(f"Created {batch_id} with {len(batch_groups)} groups")

        return batches_info