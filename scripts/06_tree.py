from ete3 import Tree, TreeStyle, NodeStyle, CircleFace
import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

# Configure headless environment
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description='Create phylogenetic tree visualization')
    parser.add_argument('--input-a3m', required=True, type=Path, 
                      help='Input A3M file for clustering')
    parser.add_argument('--output-dir', required=True, type=Path,
                      help='Output directory for results')
    parser.add_argument('--state-a3ms', required=True, nargs='+', type=Path,
                      help='List of A3M files containing sequences for each state')
    parser.add_argument('--state-colors', required=True, nargs='+', type=str,
                      help='List of colors for each state')
    parser.add_argument('--state-names', required=True, nargs='+', type=str,
                      help='List of names for each state')
    parser.add_argument('--plot-prefix', required=True, type=str,
                      help='Prefix for output plot files')
    parser.add_argument('--min-seq-id', type=float, default=0.7,
                      help='Minimum sequence identity threshold for MMseqs2 clustering (default: 0.7)')
    parser.add_argument('--skip-clustering', action='store_true',
                      help='Skip MMseqs2 clustering step')
    args = parser.parse_args()
    
    # Validate lists have same length
    if not (len(args.state_a3ms) == len(args.state_colors) == len(args.state_names)):
        parser.error("Number of state A3Ms, colors and names must match")
        
    return args

def read_fasta_sequences(filepath: Path) -> Dict[str, str]:
    """Read sequences from FASTA/A3M file into dictionary."""
    sequences = {}
    current_header = None
    current_seq = []
    
    with open(filepath) as f:
        for line in f:
            if line.startswith('>') and not line.startswith(('#', '>query')):
                if current_header:
                    sequences[current_header] = ''.join(s for s in ''.join(current_seq) 
                                                      if s.isupper() or s == '-')
                current_header = line.strip().split()[0].replace(':', '_')
                current_seq = []
            elif current_header and not line.startswith('#'):
                current_seq.append(line.strip())
                
    if current_header:
        sequences[current_header] = ''.join(s for s in ''.join(current_seq) 
                                          if s.isupper() or s == '-')
    return sequences

def run_mmseqs_clustering(min_seq_id, input_a3m: Path, output_dir: Path):
    """Run MMseqs2 clustering on input sequences."""
    mmseqs_cmd = [
        "/fs/ess/PAA0203/xing244/packages/mmseqs/bin/mmseqs",
        "easy-cluster", 
        str(input_a3m),
        str(output_dir / "result"),
        str(output_dir / "tmp"),
        "--min-seq-id", str(min_seq_id)
    ]
    subprocess.run(mmseqs_cmd, check=True)

def run_fasttree(input_fasta: Path, output_tree: Path):
    """Generate phylogenetic tree using FastTree."""
    fasttree_cmd = [
        "/fs/ess/PAA0203/xing244/packages/FastTree/FastTreeMP",
        str(input_fasta)
    ]
    with open(output_tree, "w") as f:
        subprocess.run(fasttree_cmd, stdout=f, check=True)

def read_a3m_headers(filepath: Path) -> Set[str]:
    """Extract sequence headers from A3M file."""
    headers = set()
    with open(filepath) as f:
        for line in f:
            if line.startswith('>') and not line.startswith(('#', '>query')):
                headers.add(line[1:].strip().split()[0].replace(':', '_'))
    return headers

def create_tree_visualization(tree_file: Path, state_seqs: Set[str], state_color: str, output_plot: Path):
    """Create and save tree visualization."""
    # Load tree and configure style
    tree = Tree(str(tree_file))
    ts = TreeStyle()
    ts.mode = "c"  # Circular layout
    ts.show_leaf_name = False
    ts.show_branch_length = False 
    ts.show_branch_support = False

    # Style nodes and branches
    for node in tree.traverse():
        # Set default style for all nodes
        style = NodeStyle()
        style["hz_line_width"] = 2  # Set horizontal branch thickness
        style["vt_line_width"] = 2  # Set vertical branch thickness

        if node.name in state_seqs:
            style["bgcolor"] = state_color
        
        node.set_style(style)

    # Save visualization with high resolution
    tree.render(str(output_plot), tree_style=ts, w=8, h=8, units="in", dpi=900)

def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")
    
    # Run clustering if not skipped
    if not args.skip_clustering:
        print("Running MMseqs2 clustering...")
        run_mmseqs_clustering(args.min_seq_id, args.input_a3m, args.output_dir)
        print("Completed MMseqs2 clustering")
    
    # Combine sequences
    print("Combining sequences...")
    combined_seqs = {}
    if not args.skip_clustering:
        combined_seqs.update(read_fasta_sequences(args.output_dir / "result_rep_seq.fasta"))

    elif args.skip_clustering:
        combined_seqs.update(read_fasta_sequences(args.input_a3m))



    for state_a3m in args.state_a3ms:
        combined_seqs.update(read_fasta_sequences(state_a3m))
    
    # Write combined sequences
    combined_fasta = args.output_dir / "combined_rep_seq.fasta"
    with open(combined_fasta, 'w') as f:
        for header, seq in combined_seqs.items():
            f.write(f"{header}\n{seq}\n")
    
    # Generate tree once
    print("Running FastTree analysis...")
    tree_file = args.output_dir / "tree"
    run_fasttree(combined_fasta, tree_file)
    print("Completed FastTree analysis")
    
    # Create visualization for each state
    print("Creating tree visualizations...")
    for state_a3m, state_color, state_name in zip(args.state_a3ms, args.state_colors, args.state_names):
        state_seqs = read_a3m_headers(state_a3m)
        plot_name = f"{args.plot_prefix}_{state_name}_tree_plot.png"
        output_plot = args.output_dir / plot_name
        
        print(f"Creating visualization for state {state_name}...")
        create_tree_visualization(tree_file, state_seqs, state_color, output_plot)
        print(f"Tree visualization saved as '{output_plot}'")

if __name__ == "__main__":
    main()
