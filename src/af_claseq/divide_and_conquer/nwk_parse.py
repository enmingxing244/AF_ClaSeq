#!/usr/bin/env python3
"""
Parse Newick tree from FastTree2 and split a3m alignment into clade-based subsets
"""

import os
from Bio import SeqIO, AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from ete3 import Tree
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import argparse


def read_a3m(filename):
    """Read a3m alignment file"""
    # a3m is similar to FASTA format
    records = {}
    with open(filename, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    records[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]  # Get ID without description
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            records[current_id] = ''.join(current_seq)
    return records


def write_a3m(records, filename):
    """Write sequences to a3m format file"""
    with open(filename, 'w') as f:
        for seq_id, seq in records.items():
            f.write(f'>{seq_id}\n')
            # Write sequence in chunks of 80 characters
            f.write(seq + '\n')


def get_distance_matrix(tree):
    """Calculate pairwise distances between all leaves in the tree"""
    leaves = tree.get_leaves()
    n = len(leaves)
    dist_matrix = np.zeros((n, n))
    
    for i, leaf1 in enumerate(leaves):
        for j, leaf2 in enumerate(leaves):
            if i != j:
                dist_matrix[i, j] = tree.get_distance(leaf1, leaf2)
    
    return dist_matrix, [leaf.name for leaf in leaves]


def cluster_by_distance(tree, n_clusters=None, distance_threshold=None):
    """
    Cluster tree leaves by distance using hierarchical clustering
    
    Args:
        tree: ete3 Tree object
        n_clusters: Number of clusters (if specified)
        distance_threshold: Distance threshold for clustering (if specified)
    
    Returns:
        Dictionary mapping cluster_id to list of sequence IDs
    """
    # Get distance matrix
    dist_matrix, leaf_names = get_distance_matrix(tree)
    
    # Convert to condensed distance matrix for scipy
    condensed_dist = squareform(dist_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Get cluster assignments
    if n_clusters:
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    elif distance_threshold:
        clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    else:
        # Default: use 5 clusters
        clusters = fcluster(linkage_matrix, 5, criterion='maxclust')
    
    # Group sequences by cluster
    cluster_dict = {}
    for seq_id, cluster_id in zip(leaf_names, clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(seq_id)
    
    return cluster_dict


def get_monophyletic_clades(tree, min_size=3, max_size=None):
    """
    Find monophyletic clades in the tree with recursive subdivision
    
    Args:
        tree: ete3 Tree object
        min_size: Minimum number of sequences in a clade
        max_size: Maximum number of sequences in a clade
    
    Returns:
        List of clades (each clade is a list of sequence IDs)
    """
    clades = []
    total_leaves = len(tree.get_leaves())
    
    if max_size is None:
        max_size = total_leaves // 2  # Don't return clades larger than half the tree
    
    def recursively_subdivide(node, depth=0):
        """
        Recursively subdivide a node if it's too large
        Returns True if this node was successfully processed
        """
        leaves = node.get_leaves()
        n_leaves = len(leaves)
        leaf_names = [leaf.name for leaf in leaves]
        
        # If node is small enough and meets minimum size, add it as a clade
        if n_leaves <= max_size:
            if n_leaves >= min_size:
                clades.append(leaf_names)
                return True
            elif n_leaves > 0:
                # Too small for a clade, but we'll collect these later
                return False
        
        # Node is too large, need to subdivide
        if len(node.children) == 0:
            # This is a leaf node, shouldn't happen but handle it
            if n_leaves >= min_size:
                clades.append(leaf_names)
            return True
        
        # Try to subdivide using children
        successfully_divided = True
        unclaimed_leaves = set(leaf_names)
        
        # First, recursively process all children
        for child in node.children:
            child_leaves = [leaf.name for leaf in child.get_leaves()]
            if len(child_leaves) > 0:
                if recursively_subdivide(child, depth + 1):
                    # Remove successfully processed leaves
                    for leaf in child_leaves:
                        unclaimed_leaves.discard(leaf)
        
        # If we have unclaimed leaves that together meet the minimum size,
        # try to group them into clades
        if unclaimed_leaves and len(unclaimed_leaves) >= min_size:
            # Group unclaimed leaves by their immediate parent
            parent_groups = {}
            for child in node.children:
                child_leaf_names = set([leaf.name for leaf in child.get_leaves()])
                group = list(child_leaf_names.intersection(unclaimed_leaves))
                if group:
                    parent_groups[child] = group
            
            # Add groups that meet size requirements
            for parent, group in parent_groups.items():
                if len(group) >= min_size:
                    clades.append(group)
                    for leaf in group:
                        unclaimed_leaves.discard(leaf)
            
            # If still have unclaimed leaves, group them together if they meet min_size
            unclaimed_list = list(unclaimed_leaves)
            if len(unclaimed_list) >= min_size:
                # If this group is still too large, split it arbitrarily
                while len(unclaimed_list) > max_size:
                    clades.append(unclaimed_list[:max_size])
                    unclaimed_list = unclaimed_list[max_size:]
                if len(unclaimed_list) >= min_size:
                    clades.append(unclaimed_list)
        
        return True
    
    # Start recursive subdivision from root
    recursively_subdivide(tree)
    
    # Ensure no duplicates
    seen = set()
    unique_clades = []
    for clade in clades:
        clade_set = frozenset(clade)
        if clade_set not in seen:
            seen.add(clade_set)
            unique_clades.append(clade)
    
    return unique_clades


def get_monophyletic_clades_with_stats(tree, min_size=3, max_size=100):
    """
    Find monophyletic clades with detailed statistics
    
    Args:
        tree: ete3 Tree object
        min_size: Minimum number of sequences in a clade
        max_size: Maximum number of sequences in a clade
    
    Returns:
        Tuple of (clades, stats_dict)
    """
    clades = []
    stats = {
        'total_sequences': len(tree.get_leaves()),
        'clades_found': 0,
        'sequences_assigned': 0,
        'clade_sizes': [],
        'subdivisions_made': 0
    }
    
    def recursively_subdivide_with_stats(node, depth=0):
        """
        Recursively subdivide with statistics tracking
        """
        leaves = node.get_leaves()
        n_leaves = len(leaves)
        leaf_names = [leaf.name for leaf in leaves]
        
        print(f"{'  ' * depth}Processing node with {n_leaves} sequences")
        
        # If node is small enough, add as clade
        if n_leaves <= max_size:
            if n_leaves >= min_size:
                clades.append(leaf_names)
                stats['clade_sizes'].append(n_leaves)
                print(f"{'  ' * depth}✓ Added clade with {n_leaves} sequences")
                return True
            else:
                print(f"{'  ' * depth}✗ Too small ({n_leaves} < {min_size})")
                return False
        
        # Node too large, subdivide
        print(f"{'  ' * depth}↓ Too large ({n_leaves} > {max_size}), subdividing...")
        stats['subdivisions_made'] += 1
        
        if len(node.children) == 0:
            # Leaf node - shouldn't happen
            clades.append(leaf_names)
            stats['clade_sizes'].append(n_leaves)
            return True
        
        # Process all children
        for i, child in enumerate(node.children):
            child_size = len(child.get_leaves())
            if child_size > 0:
                print(f"{'  ' * depth}  Child {i+1}/{len(node.children)}: {child_size} sequences")
                recursively_subdivide_with_stats(child, depth + 1)
        
        return True
    
    # Start from root
    print(f"\nStarting recursive subdivision (max_size={max_size}, min_size={min_size})...")
    recursively_subdivide_with_stats(tree)
    
    # Remove duplicates
    seen = set()
    unique_clades = []
    for clade in clades:
        clade_set = frozenset(clade)
        if clade_set not in seen:
            seen.add(clade_set)
            unique_clades.append(clade)
    
    # Update stats
    stats['clades_found'] = len(unique_clades)
    stats['sequences_assigned'] = sum(len(c) for c in unique_clades)
    
    return unique_clades, stats


def manual_clade_selection(tree, alignment):
    """
    Interactive clade selection using tree visualization
    """
    print("\nTree structure:")
    print(tree.get_ascii(show_internal=True))
    
    clades = []
    while True:
        print("\nEnter sequence IDs for a clade (comma-separated), or 'done' to finish:")
        user_input = input().strip()
        
        if user_input.lower() == 'done':
            break
        
        seq_ids = [s.strip() for s in user_input.split(',')]
        # Validate sequence IDs
        valid_ids = []
        for seq_id in seq_ids:
            if seq_id in alignment:
                valid_ids.append(seq_id)
            else:
                print(f"Warning: {seq_id} not found in alignment")
        
        if valid_ids:
            clades.append(valid_ids)
            print(f"Added clade with {len(valid_ids)} sequences")
    
    return clades


def split_alignment_by_clades(tree_file, a3m_file, output_dir="clades", 
                             method="auto", n_clusters=None, 
                             distance_threshold=None, min_clade_size=3,
                             max_clade_size=None, verbose=False):
    """
    Main function to split alignment based on tree clades
    
    Args:
        tree_file: Path to Newick tree file
        a3m_file: Path to a3m alignment file
        output_dir: Directory for output files
        method: "auto", "distance", or "manual"
        n_clusters: Number of clusters for distance-based clustering
        distance_threshold: Distance threshold for clustering
        min_clade_size: Minimum size for automatic clade detection
        max_clade_size: Maximum size for automatic clade detection (will subdivide larger clades)
        verbose: Print detailed progress information
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read tree
    print(f"Reading tree from {tree_file}...")
    tree = Tree(tree_file)
    
    # Read alignment
    print(f"Reading alignment from {a3m_file}...")
    alignment = read_a3m(a3m_file)
    print(f"Total sequences in alignment: {len(alignment)}")
    
    # Get clades based on method
    if method == "manual":
        clades = manual_clade_selection(tree, alignment)
    elif method == "distance":
        print("Clustering by distance...")
        cluster_dict = cluster_by_distance(tree, n_clusters, distance_threshold)
        clades = list(cluster_dict.values())
    else:  # auto
        print("Finding monophyletic clades...")
        if verbose:
            clades, stats = get_monophyletic_clades_with_stats(
                tree, min_size=min_clade_size, max_size=max_clade_size
            )
            print(f"\nClade detection statistics:")
            print(f"  Total sequences: {stats['total_sequences']}")
            print(f"  Clades found: {stats['clades_found']}")
            print(f"  Sequences assigned: {stats['sequences_assigned']}")
            print(f"  Subdivisions made: {stats['subdivisions_made']}")
            if stats['clade_sizes']:
                print(f"  Clade sizes: min={min(stats['clade_sizes'])}, "
                      f"max={max(stats['clade_sizes'])}, "
                      f"mean={sum(stats['clade_sizes'])/len(stats['clade_sizes']):.1f}")
        else:
            clades = get_monophyletic_clades(
                tree, min_size=min_clade_size, max_size=max_clade_size
            )
    
    if not clades:
        print("No clades found!")
        return
    
    print(f"\nFound {len(clades)} clades")
    
    # Sort clades by size for better organization
    clades.sort(key=len, reverse=True)
    
    # Split alignment
    clade_summary = []
    for i, clade_seqs in enumerate(clades, 1):
        # Filter alignment for this clade
        clade_alignment = {}
        missing_seqs = []
        
        for seq_id in clade_seqs:
            if seq_id in alignment:
                clade_alignment[seq_id] = alignment[seq_id]
            else:
                missing_seqs.append(seq_id)
        
        if missing_seqs and verbose:
            print(f"Warning: {len(missing_seqs)} sequences from clade {i} not found in alignment")
        
        if clade_alignment:
            output_file = os.path.join(output_dir, f"clade_{i:03d}.a3m")
            write_a3m(clade_alignment, output_file)
            clade_summary.append((i, len(clade_alignment), output_file))
            if verbose:
                print(f"Clade {i}: {len(clade_alignment)} sequences -> {output_file}")
    
    # Handle unclustered sequences
    all_clustered = set()
    for clade in clades:
        all_clustered.update(clade)
    
    unclustered = set(alignment.keys()) - all_clustered
    if unclustered:
        unclustered_alignment = {seq_id: alignment[seq_id] for seq_id in unclustered}
        output_file = os.path.join(output_dir, "unclustered.a3m")
        write_a3m(unclustered_alignment, output_file)
        print(f"Unclustered: {len(unclustered)} sequences -> {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for clade_id, size, filepath in clade_summary:
        print(f"Clade {clade_id:3d}: {size:5d} sequences")
    if unclustered:
        print(f"Unclustered: {len(unclustered):5d} sequences")
    print(f"\nTotal clades: {len(clades)}")
    print(f"Coverage: {len(all_clustered)}/{len(alignment)} sequences assigned to clades")


def main():
    parser = argparse.ArgumentParser(
        description='Split a3m alignment by phylogenetic clades with recursive subdivision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect clades with max 100 sequences per clade
  %(prog)s tree.nwk alignment.a3m --max-size 100
  
  # Set both min and max clade sizes
  %(prog)s tree.nwk alignment.a3m --min-size 5 --max-size 50
  
  # Distance-based clustering with 10 clusters
  %(prog)s tree.nwk alignment.a3m -m distance -n 10
  
  # Verbose output to see subdivision process
  %(prog)s tree.nwk alignment.a3m --max-size 100 -v
        """
    )
    parser.add_argument('tree_file', help='Newick tree file from FastTree2')
    parser.add_argument('a3m_file', help='a3m alignment file')
    parser.add_argument('-o', '--output-dir', default='clades', 
                       help='Output directory for subset files (default: clades)')
    parser.add_argument('-m', '--method', choices=['auto', 'distance', 'manual'], 
                       default='auto',
                       help='Method for identifying clades (default: auto)')
    parser.add_argument('-n', '--n-clusters', type=int, 
                       help='Number of clusters for distance method')
    parser.add_argument('-d', '--distance-threshold', type=float,
                       help='Distance threshold for clustering')
    parser.add_argument('--min-size', type=int, default=3,
                       help='Minimum clade size (default: 3)')
    parser.add_argument('--max-size', type=int, default=None,
                       help='Maximum clade size - larger clades will be recursively subdivided')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output showing subdivision process')
    
    args = parser.parse_args()
    
    split_alignment_by_clades(
        args.tree_file,
        args.a3m_file,
        args.output_dir,
        args.method,
        args.n_clusters,
        args.distance_threshold,
        args.min_size,
        args.max_size,
        args.verbose
    )


if __name__ == "__main__":
    main()