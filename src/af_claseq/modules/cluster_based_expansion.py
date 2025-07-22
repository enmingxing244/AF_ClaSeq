#!/usr/bin/env python3
"""
MMseqs2 cluster-based sequence expansion for hit expand pipeline.

This module provides functionality to expand sequences using pre-computed
MMseqs2 clustering results instead of BLOSUM62 similarity search.
"""

import logging
from pathlib import Path
from typing import Dict, Set, Optional, Any, Tuple
from collections import defaultdict

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.sequence_processing import A3MParser


class ClusterExpansionError(Exception):
    """Raised when cluster-based expansion fails."""
    pass


class ClusterBasedExpansion:
    """
    MMseqs2 cluster-based sequence expansion for hit expand pipeline.
    
    Uses pre-computed MMseqs2 clustering results to expand sequences
    by collecting all members of the same clusters.
    """
    
    def __init__(self, cluster_file: Path, source_msa: Path):
        """
        Initialize with cluster file and source MSA.
        
        Args:
            cluster_file: Path to clustered_cluster.tsv from MMseqs2
            source_msa: Path to source A3M file with all sequences
        """
        self.cluster_file = Path(cluster_file)
        self.source_msa = Path(source_msa)
        self.seq_to_representative = {}
        self.representative_to_members = defaultdict(set)
        self.source_sequences = {}
        self.original_headers = {}  # Maps clean IDs to original headers
        self.logger = get_logger(__name__)
        
        self._validate_inputs()
        self._load_cluster_mappings()
        self._load_source_sequences()
    
    def _validate_inputs(self):
        """Validate that input files exist."""
        if not self.cluster_file.exists():
            raise ClusterExpansionError(f"Cluster file not found: {self.cluster_file}")
        if not self.source_msa.exists():
            raise ClusterExpansionError(f"Source MSA file not found: {self.source_msa}")
    
    def _clean_header(self, header: str) -> str:
        """
        Clean header for matching with TSV file entries.
        
        Logic:
        1. Remove ">" if present
        2. Split by any whitespace (space or tab) - take first part
        3. That's the header for TSV matching
        
        Examples:
        - ">UniRef100_A0A4U5P8Y6 70\t0.278..." -> "UniRef100_A0A4U5P8Y6"
        - ">ERR1719483_140725 187\t0.307..." -> "ERR1719483_140725"  
        - ">seq123\tdescription" -> "seq123"
        - ">seq123 description" -> "seq123"
        
        Args:
            header: Raw header string
            
        Returns:
            Cleaned header ID for TSV matching
        """
        # Remove ">" if present
        cleaned = header.lstrip('>')
        
        # Split by any whitespace (space, tab, etc.) and take first part
        cleaned = cleaned.split()[0]
        
        return cleaned.strip()
    
    def _load_cluster_mappings(self):
        """Build bidirectional cluster mappings from TSV file."""
        try:
            self.logger.info(f"Loading cluster mappings from {self.cluster_file}")
            
            with open(self.cluster_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        representative = parts[0].strip()
                        member = parts[1].strip()
                        
                        # Build mappings
                        self.seq_to_representative[member] = representative
                        self.representative_to_members[representative].add(member)
                    else:
                        self.logger.warning(f"Skipping malformed line {line_num}: {line}")
                        
            self.logger.info(f"Loaded {len(self.seq_to_representative)} sequence-to-cluster mappings")
            self.logger.info(f"Found {len(self.representative_to_members)} clusters")
            
            # Log some statistics
            cluster_sizes = [len(members) for members in self.representative_to_members.values()]
            if cluster_sizes:
                self.logger.info(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
                               f"avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
            
        except Exception as e:
            self.logger.error(f"Error loading cluster mappings: {e}")
            raise ClusterExpansionError(f"Failed to load cluster mappings: {e}")
    
    def _load_source_sequences(self):
        """Load all sequences from source MSA with flexible header parsing."""
        try:
            self.logger.info(f"Loading sequences from {self.source_msa}")
            
            parser = A3MParser(strict_validation=False)
            raw_sequences = parser.parse_file(self.source_msa)
            
            # Create mapping with cleaned headers
            for header, sequence in raw_sequences.items():
                clean_id = self._clean_header(header)
                
                # Store the sequence
                self.source_sequences[clean_id] = sequence
                
                # Keep track of original headers for clean IDs
                if clean_id not in self.original_headers:
                    self.original_headers[clean_id] = header
                else:
                    # If we have duplicate clean IDs, log a warning
                    self.logger.debug(f"Duplicate clean ID '{clean_id}' found. "
                                    f"Original: {self.original_headers[clean_id]}, "
                                    f"New: {header}")
                
            self.logger.info(f"Loaded {len(raw_sequences)} sequences from source MSA")
            self.logger.info(f"Created {len(self.source_sequences)} unique clean IDs")
            
        except Exception as e:
            self.logger.error(f"Error loading source sequences: {e}")
            raise ClusterExpansionError(f"Failed to load source sequences: {e}")
    
    def expand_by_clusters(self, 
                          good_sequences: Dict[str, str],
                          exclude_sequences: Optional[Set[str]] = None,
                          output_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Expand sequences by collecting all cluster members.
        
        Args:
            good_sequences: Dict of good sequences to expand {header: sequence}
            exclude_sequences: Optional set of sequence IDs to exclude
            output_dir: Optional output directory for detailed logs
            
        Returns:
            Dict of expanded sequences including all cluster members
        """
        expanded_sequences = {}
        exclude_sequences = exclude_sequences or set()
        
        # Clean the exclusion set
        clean_exclude = {self._clean_header(seq_id) for seq_id in exclude_sequences}
        
        # Prepare detailed logging
        expansion_details = []
        good_seq_details = []
        
        statistics = {
            'input_sequences': len(good_sequences),
            'clusters_found': 0,
            'sequences_not_in_clusters': 0,
            'total_expanded': 0,
            'excluded_count': 0,
            'sequences_per_cluster': [],
            'missing_sequences': []
        }
        
        # Track which sequences we've already added to avoid duplicates
        added_clean_ids = set()
        
        # Process each good sequence
        for header, sequence in good_sequences.items():
            # Always add the good sequence itself
            expanded_sequences[header] = sequence
            clean_id = self._clean_header(header)
            added_clean_ids.add(clean_id)
            
            # Record good sequence details
            good_seq_details.append({
                'original_header': header,
                'cleaned_id': clean_id,
                'sequence_length': len(sequence)
            })
            
            # Find cluster representative
            representative = None
            if clean_id in self.seq_to_representative:
                representative = self.seq_to_representative[clean_id]
            elif clean_id in self.representative_to_members:
                # It's already a representative
                representative = clean_id
            else:
                self.logger.debug(f"No cluster found for sequence: {clean_id}")
                statistics['sequences_not_in_clusters'] += 1
                # Record no cluster found
                expansion_details.append({
                    'good_seq_header': header,
                    'good_seq_cleaned': clean_id,
                    'cluster_representative': 'NOT_FOUND',
                    'cluster_members': [],
                    'added_members': [],
                    'excluded_members': [],
                    'missing_members': []
                })
                continue
                
            statistics['clusters_found'] += 1
            
            # Get all cluster members
            cluster_members = self.representative_to_members.get(representative, set())
            cluster_expansion_count = 0
            
            # Track details for this expansion
            expansion_detail = {
                'good_seq_header': header,
                'good_seq_cleaned': clean_id,
                'cluster_representative': representative,
                'cluster_members': list(cluster_members),
                'added_members': [],
                'excluded_members': [],
                'missing_members': []
            }
            
            # Add all cluster members
            for member_id in cluster_members:
                # Skip if already added
                if member_id in added_clean_ids:
                    continue
                    
                # Skip if in exclusion list
                if member_id in clean_exclude:
                    statistics['excluded_count'] += 1
                    expansion_detail['excluded_members'].append(member_id)
                    continue
                
                # Try to find sequence
                if member_id in self.source_sequences:
                    # Use original header if available, otherwise create a simple one
                    original_header = self.original_headers.get(member_id, f">{member_id}")
                    expanded_sequences[original_header] = self.source_sequences[member_id]
                    added_clean_ids.add(member_id)
                    cluster_expansion_count += 1
                    expansion_detail['added_members'].append(member_id)
                else:
                    self.logger.debug(f"Could not find sequence for cluster member: {member_id}")
                    statistics['missing_sequences'].append(member_id)
                    expansion_detail['missing_members'].append(member_id)
            
            expansion_details.append(expansion_detail)
            statistics['sequences_per_cluster'].append(cluster_expansion_count)
            
        statistics['total_expanded'] = len(expanded_sequences)
        self.expansion_statistics = statistics
        
        # Write detailed output files if output_dir is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Write good sequences details
            good_seq_file = output_dir / "good_sequences_details.txt"
            with open(good_seq_file, 'w') as f:
                f.write("GOOD SEQUENCES USED FOR CLUSTER EXPANSION\n")
                f.write("=" * 50 + "\n\n")
                for i, detail in enumerate(good_seq_details, 1):
                    f.write(f"{i}. Original Header: {detail['original_header']}\n")
                    f.write(f"   Cleaned ID: {detail['cleaned_id']}\n")
                    f.write(f"   Sequence Length: {detail['sequence_length']}\n\n")
            
            # Write cluster expansion details
            expansion_file = output_dir / "cluster_expansion_details.txt"
            with open(expansion_file, 'w') as f:
                f.write("CLUSTER EXPANSION DETAILS\n")
                f.write("=" * 50 + "\n\n")
                for i, detail in enumerate(expansion_details, 1):
                    f.write(f"{i}. Good Sequence: {detail['good_seq_cleaned']}\n")
                    f.write(f"   Original Header: {detail['good_seq_header']}\n")
                    f.write(f"   Cluster Representative: {detail['cluster_representative']}\n")
                    f.write(f"   Total Cluster Members: {len(detail['cluster_members'])}\n")
                    f.write(f"   Added Members: {len(detail['added_members'])}\n")
                    f.write(f"   Excluded Members: {len(detail['excluded_members'])}\n")
                    f.write(f"   Missing Members: {len(detail['missing_members'])}\n")
                    
                    if detail['cluster_members']:
                        f.write(f"   All Cluster Members: {', '.join(detail['cluster_members'][:10])}")
                        if len(detail['cluster_members']) > 10:
                            f.write(f" ... and {len(detail['cluster_members']) - 10} more")
                        f.write("\n")
                    
                    if detail['added_members']:
                        f.write(f"   Added to Expansion: {', '.join(detail['added_members'][:5])}")
                        if len(detail['added_members']) > 5:
                            f.write(f" ... and {len(detail['added_members']) - 5} more")
                        f.write("\n")
                    
                    if detail['excluded_members']:
                        f.write(f"   Excluded: {', '.join(detail['excluded_members'])}\n")
                    
                    if detail['missing_members']:
                        f.write(f"   Missing in Source: {', '.join(detail['missing_members'])}\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Detailed expansion files written to: {output_dir}")
            self.logger.info(f"  - Good sequences: {good_seq_file}")
            self.logger.info(f"  - Expansion details: {expansion_file}")
        
        self.logger.info(f"Expansion complete: {len(good_sequences)} -> {len(expanded_sequences)} sequences")
        self.logger.info(f"Found {statistics['clusters_found']} clusters, "
                        f"{statistics['sequences_not_in_clusters']} sequences not in clusters")
        if statistics['excluded_count'] > 0:
            self.logger.info(f"Excluded {statistics['excluded_count']} sequences")
        if statistics['missing_sequences']:
            self.logger.warning(f"Could not find {len(statistics['missing_sequences'])} sequences in source MSA")
        
        return expanded_sequences
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the expansion process.
        
        Returns:
            Dictionary with expansion statistics
        """
        if hasattr(self, 'expansion_statistics'):
            stats = self.expansion_statistics.copy()
            
            # Calculate additional statistics
            if stats['sequences_per_cluster']:
                cluster_sizes = stats['sequences_per_cluster']
                stats['avg_sequences_per_cluster'] = sum(cluster_sizes) / len(cluster_sizes)
                stats['max_cluster_size'] = max(cluster_sizes)
                stats['min_cluster_size'] = min(cluster_sizes)
                stats['expansion_ratio'] = stats['total_expanded'] / stats['input_sequences'] if stats['input_sequences'] > 0 else 0
            
            # Remove internal details from public stats
            stats.pop('missing_sequences', None)
            
            return stats
        
        return {
            'error': 'No expansion has been performed yet'
        }
    
    def get_cluster_info(self, sequence_id: str) -> Dict[str, Any]:
        """
        Get cluster information for a specific sequence.
        
        Args:
            sequence_id: Sequence ID (will be cleaned)
            
        Returns:
            Dictionary with cluster information
        """
        clean_id = self._clean_header(sequence_id)
        
        info = {
            'sequence_id': sequence_id,
            'clean_id': clean_id,
            'in_cluster': False,
            'is_representative': False,
            'cluster_representative': None,
            'cluster_size': 0,
            'cluster_members': []
        }
        
        # Check if it's in a cluster
        if clean_id in self.seq_to_representative:
            representative = self.seq_to_representative[clean_id]
            info['in_cluster'] = True
            info['cluster_representative'] = representative
            info['cluster_members'] = list(self.representative_to_members.get(representative, set()))
            info['cluster_size'] = len(info['cluster_members'])
        elif clean_id in self.representative_to_members:
            # It's a representative
            info['in_cluster'] = True
            info['is_representative'] = True
            info['cluster_representative'] = clean_id
            info['cluster_members'] = list(self.representative_to_members[clean_id])
            info['cluster_size'] = len(info['cluster_members'])
        
        return info