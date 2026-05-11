#!/usr/bin/env python3
"""
Parse Newick tree from FastTree2 and split a3m alignment into clade-based subsets
"""

import os
from ete3 import Tree
import argparse

from af_claseq.utils.sequence_processing import write_a3m, read_a3m_to_dict

def get_monophyletic_clades(tree, min_size=3, max_size=None):
    """
    Distance-guided phylogenetic splitting with adjacency-based merging.
    Pure data processing of FastTree .nwk file using branch distances.

    Args:
        tree: ete3 Tree object
        min_size: Minimum number of sequences in a clade
        max_size: Maximum number of sequences in a clade

    Returns:
        List of clades (each clade is a list of sequence IDs)
    """
    clades = []
    small_nodes = []  # Store nodes that are too small for merging
    processed_nodes = set()  # Track nodes that have been processed

    total_leaves = len(tree.get_leaves())
    if max_size is None:
        max_size = total_leaves // 2

    def process_node_by_distance(node):
        """Process node using .nwk branch distances for optimal splitting"""
        if node in processed_nodes:
            return

        leaves = node.get_leaves()
        size = len(leaves)
        leaf_names = [leaf.name for leaf in leaves]

        if min_size <= size <= max_size:
            # Perfect size - immediate collection
            clades.append(leaf_names)
            processed_nodes.add(node)

        elif size > max_size:
            # Too large - go deeper using .nwk distances
            # Sort children by branch distance for optimal traversal
            if node.children:
                children_with_distance = [(child, getattr(child, 'dist', 0)) for child in node.children]
                children_sorted = sorted(children_with_distance, key=lambda x: x[1])

                for child, _ in children_sorted:
                    process_node_by_distance(child)
            processed_nodes.add(node)

        else:  # size < min_size
            # Too small - store for adjacency merging (do NOT add to processed_nodes
            # yet — merge_small_clades needs to find merge candidates among unprocessed nodes)
            small_nodes.append(node)

    def calculate_sibling_distance(node1, node2):
        """Calculate phylogenetic distance between sibling nodes"""
        # Use branch distances from .nwk file
        dist1 = getattr(node1, 'dist', 0)
        dist2 = getattr(node2, 'dist', 0)
        return abs(dist1 - dist2)

    def _find_clade_for_node(node):
        """Find the existing clade list that contains this node's leaves, if any."""
        leaf_names = set(leaf.name for leaf in node.get_leaves())
        for i, clade in enumerate(clades):
            if leaf_names == set(clade):
                return i
        return None

    def find_best_merge_candidate(small_node):
        """Find phylogenetically closest same-level sibling for safe merging.

        Considers ALL siblings (including already-emitted clades) as merge targets,
        so small nodes can be absorbed into adjacent normal-sized clades.
        """
        parent = small_node.up
        if not parent:
            return None

        small_size = len(small_node.get_leaves())
        siblings = [child for child in parent.children if child != small_node]

        valid_candidates = []
        for sibling in siblings:
            sibling_size = len(sibling.get_leaves())
            total_size = small_size + sibling_size

            if total_size <= max_size:
                phylo_distance = calculate_sibling_distance(small_node, sibling)
                valid_candidates.append((sibling, sibling_size, phylo_distance))

        if not valid_candidates:
            return None

        best_candidate = min(valid_candidates, key=lambda x: x[2])
        return best_candidate[0]

    def merge_small_clades():
        """Merge small clades with phylogenetically adjacent ones"""
        for small_node in small_nodes:
            if small_node in processed_nodes:
                continue

            merge_candidate = find_best_merge_candidate(small_node)
            processed_nodes.add(small_node)

            if merge_candidate:
                small_seqs = [leaf.name for leaf in small_node.get_leaves()]
                candidate_seqs = [leaf.name for leaf in merge_candidate.get_leaves()]
                merged_seqs = small_seqs + candidate_seqs

                # If merge candidate was already emitted as a clade, replace it
                existing_idx = _find_clade_for_node(merge_candidate)
                if existing_idx is not None:
                    clades[existing_idx] = merged_seqs
                else:
                    clades.append(merged_seqs)
                processed_nodes.add(merge_candidate)
            else:
                small_seqs = [leaf.name for leaf in small_node.get_leaves()]
                parent = small_node.up
                if parent:
                    for sibling in parent.children:
                        if sibling != small_node and sibling not in processed_nodes:
                            sibling_seqs = [leaf.name for leaf in sibling.get_leaves()]
                            forced_merge = small_seqs + sibling_seqs
                            clades.append(forced_merge)
                            processed_nodes.add(sibling)
                            break
                    else:
                        clades.append(small_seqs)

    # Execute the distance-guided splitting algorithm
    process_node_by_distance(tree)
    merge_small_clades()

    return clades


def get_monophyletic_clades_with_stats(tree, min_size=3, max_size=100):
    """
    Distance-guided phylogenetic splitting with detailed statistics

    Args:
        tree: ete3 Tree object
        min_size: Minimum number of sequences in a clade
        max_size: Maximum number of sequences in a clade

    Returns:
        Tuple of (clades, stats_dict)
    """
    stats = {
        'total_sequences': len(tree.get_leaves()),
        'clades_found': 0,
        'sequences_assigned': 0,
        'clade_sizes': [],
        'subdivisions_made': 0,
        'merges_performed': 0,
        'forced_merges': 0
    }

    print(f"\nStarting distance-guided splitting (max_size={max_size}, min_size={min_size})...")
    print(f"Total sequences to process: {stats['total_sequences']}")

    clades = []
    small_nodes = []
    processed_nodes = set()

    def process_node_with_stats(node, depth=0):
        """Process node with statistics tracking"""
        if node in processed_nodes:
            return

        leaves = node.get_leaves()
        size = len(leaves)
        leaf_names = [leaf.name for leaf in leaves]

        print(f"{'  ' * depth}Processing node with {size} sequences")

        if min_size <= size <= max_size:
            # Perfect size
            clades.append(leaf_names)
            stats['clade_sizes'].append(size)
            processed_nodes.add(node)
            print(f"{'  ' * depth}✓ Added clade with {size} sequences")

        elif size > max_size:
            # Too large - subdivide
            print(f"{'  ' * depth}↓ Too large ({size} > {max_size}), subdividing...")
            stats['subdivisions_made'] += 1

            if node.children:
                children_with_distance = [(child, getattr(child, 'dist', 0)) for child in node.children]
                children_sorted = sorted(children_with_distance, key=lambda x: x[1])

                for i, (child, dist) in enumerate(children_sorted):
                    child_size = len(child.get_leaves())
                    print(f"{'  ' * depth}  Child {i+1}/{len(node.children)}: {child_size} sequences (dist={dist:.4f})")
                    process_node_with_stats(child, depth + 1)
            processed_nodes.add(node)

        else:  # size < min_size
            # Too small — store for merging (do NOT add to processed_nodes yet)
            small_nodes.append(node)
            print(f"{'  ' * depth}◦ Too small ({size} < {min_size}), marking for merge")

    def _find_clade_for_node_stats(node):
        """Find the existing clade list that contains this node's leaves, if any."""
        leaf_names = set(leaf.name for leaf in node.get_leaves())
        for i, clade in enumerate(clades):
            if leaf_names == set(clade):
                return i
        return None

    def merge_with_stats():
        """Merge small clades with statistics tracking"""
        print(f"\nProcessing {len(small_nodes)} small nodes for merging...")

        for small_node in small_nodes:
            if small_node in processed_nodes:
                continue
            processed_nodes.add(small_node)

            small_size = len(small_node.get_leaves())
            parent = small_node.up

            if not parent:
                continue

            # Find best merge candidate — consider ALL siblings including already-emitted
            siblings = [child for child in parent.children if child != small_node]
            best_candidate = None
            best_distance = float('inf')

            for sibling in siblings:
                sibling_size = len(sibling.get_leaves())
                total_size = small_size + sibling_size

                if total_size <= max_size:
                    dist1 = getattr(small_node, 'dist', 0)
                    dist2 = getattr(sibling, 'dist', 0)
                    phylo_distance = abs(dist1 - dist2)

                    if phylo_distance < best_distance:
                        best_distance = phylo_distance
                        best_candidate = sibling

            if best_candidate:
                small_seqs = [leaf.name for leaf in small_node.get_leaves()]
                candidate_seqs = [leaf.name for leaf in best_candidate.get_leaves()]
                merged_seqs = small_seqs + candidate_seqs

                existing_idx = _find_clade_for_node_stats(best_candidate)
                if existing_idx is not None:
                    old_size = len(clades[existing_idx])
                    clades[existing_idx] = merged_seqs
                    # Update clade_sizes: remove old, add new
                    if old_size in stats['clade_sizes']:
                        stats['clade_sizes'].remove(old_size)
                    stats['clade_sizes'].append(len(merged_seqs))
                else:
                    clades.append(merged_seqs)
                    stats['clade_sizes'].append(len(merged_seqs))
                stats['merges_performed'] += 1
                processed_nodes.add(best_candidate)

                print(f"  ✓ Merged {small_size} + {len(candidate_seqs)} = {len(merged_seqs)} sequences")

            else:
                small_seqs = [leaf.name for leaf in small_node.get_leaves()]
                if parent:
                    for sibling in parent.children:
                        if sibling != small_node and sibling not in processed_nodes:
                            sibling_seqs = [leaf.name for leaf in sibling.get_leaves()]
                            forced_merge = small_seqs + sibling_seqs
                            clades.append(forced_merge)
                            stats['clade_sizes'].append(len(forced_merge))
                            stats['forced_merges'] += 1
                            processed_nodes.add(sibling)
                            print(f"  ⚠ Forced merge {small_size} + {len(sibling_seqs)} = {len(forced_merge)} sequences")
                            break
                    else:
                        clades.append(small_seqs)
                        stats['clade_sizes'].append(len(small_seqs))
                        print(f"  ⚠ Added small clade as-is: {small_size} sequences")

    # Execute algorithm with stats
    process_node_with_stats(tree)
    merge_with_stats()

    # Update final stats
    stats['clades_found'] = len(clades)
    stats['sequences_assigned'] = sum(len(c) for c in clades)

    print(f"\nDistance-guided splitting completed:")
    print(f"  Clades found: {stats['clades_found']}")
    print(f"  Sequences assigned: {stats['sequences_assigned']}/{stats['total_sequences']}")
    print(f"  Subdivisions made: {stats['subdivisions_made']}")
    print(f"  Merges performed: {stats['merges_performed']}")
    print(f"  Forced merges: {stats['forced_merges']}")
    if stats['clade_sizes']:
        print(f"  Clade sizes: min={min(stats['clade_sizes'])}, "
              f"max={max(stats['clade_sizes'])}, "
              f"mean={sum(stats['clade_sizes'])/len(stats['clade_sizes']):.1f}")

    return clades, stats




def split_alignment_by_clades(tree_file, a3m_file, output_dir="clades",
                             min_clade_size=3, max_clade_size=None, verbose=False):
    """
    Split alignment using distance-guided phylogenetic clade detection.
    Pure data processing of FastTree .nwk file with adjacency-based merging.

    Args:
        tree_file: Path to Newick tree file
        a3m_file: Path to a3m alignment file
        output_dir: Directory for output files
        min_clade_size: Minimum clade size
        max_clade_size: Maximum clade size (larger clades will be subdivided)
        verbose: Print detailed progress information
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read tree
    print(f"Reading tree from {tree_file}...")
    tree = Tree(tree_file)

    # Read alignment
    print(f"Reading alignment from {a3m_file}...")
    alignment_raw = read_a3m_to_dict(a3m_file)

    # Normalize headers by removing '>' prefix to match tree sequence names
    alignment = {}
    for header, sequence in alignment_raw.items():
        clean_header = header.lstrip('>')  # Remove '>' prefix if present
        alignment[clean_header] = sequence

    print(f"Total sequences in alignment: {len(alignment)}")

    # Get clades using distance-guided algorithm
    print("Distance-guided phylogenetic clade detection...")
    if verbose:
        clades, stats = get_monophyletic_clades_with_stats(
            tree, min_size=min_clade_size, max_size=max_clade_size
        )
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
    total_sequences_assigned = 0

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
            total_sequences_assigned += len(clade_alignment)
            if verbose:
                print(f"Clade {i}: {len(clade_alignment)} sequences -> {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("DISTANCE-GUIDED SPLITTING SUMMARY")
    print("="*60)
    for clade_id, size, filepath in clade_summary:
        print(f"Clade {clade_id:3d}: {size:5d} sequences")
    print(f"\nTotal clades: {len(clades)}")
    print(f"Complete coverage: {total_sequences_assigned}/{len(alignment)} sequences (no unclustered)")
    print(f"Average clade size: {total_sequences_assigned/len(clades):.1f} sequences")


def main():
    parser = argparse.ArgumentParser(
        description='Split a3m alignment using distance-guided phylogenetic clade detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split with max 100 sequences per clade
  %(prog)s tree.nwk alignment.a3m --max-size 100

  # Set both min and max clade sizes
  %(prog)s tree.nwk alignment.a3m --min-size 5 --max-size 50

  # Verbose output showing detailed splitting process
  %(prog)s tree.nwk alignment.a3m --max-size 100 -v
        """
    )
    parser.add_argument('tree_file', help='Newick tree file from FastTree2')
    parser.add_argument('a3m_file', help='a3m alignment file')
    parser.add_argument('-o', '--output-dir', default='clades',
                       help='Output directory for clade files (default: clades)')
    parser.add_argument('--min-size', type=int, default=3,
                       help='Minimum clade size (default: 3)')
    parser.add_argument('--max-size', type=int, default=None,
                       help='Maximum clade size - larger clades will be subdivided using branch distances')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output showing detailed splitting process with statistics')

    args = parser.parse_args()

    split_alignment_by_clades(
        args.tree_file,
        args.a3m_file,
        args.output_dir,
        args.min_size,
        args.max_size,
        args.verbose
    )


if __name__ == "__main__":
    main()