#!/fs/ess/PAA0203/xing244/.conda/envs/colabfold/bin/python
"""
PDB Structure PCA Analysis Tool

This script:
1. Reads all PDB files from a given folder (recursively in all subdirs)
2. Superimposes structures globally by CA atoms using multiprocessing
3. Performs PCA analysis on the superimposed CA coordinates
4. Outputs results and visualizations
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count

from Bio import PDB
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_logging(log_file):
    """Setup logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_ca_atoms(structure):
    """Extract CA atoms from a structure."""
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Standard residue
                    if 'CA' in residue:
                        ca_atoms.append(residue['CA'])
    return ca_atoms


def get_ca_coords(structure):
    """Get CA atom coordinates as numpy array."""
    ca_atoms = extract_ca_atoms(structure)
    coords = np.array([atom.get_coord() for atom in ca_atoms])
    return coords


def load_pdb(pdb_file):
    """Load a PDB file and return structure."""
    parser = PDB.PDBParser(QUIET=True)
    structure_id = os.path.basename(pdb_file).replace('.pdb', '')
    try:
        structure = parser.get_structure(structure_id, pdb_file)
        return structure, pdb_file
    except Exception as e:
        return None, pdb_file


def superimpose_to_reference(args):
    """
    Superimpose a mobile structure to reference CA coordinates using Kabsch algorithm.

    Args:
        args: tuple of (pdb_file, ref_ca_coords)

    Returns:
        tuple: (pdb_file, superimposed_coords, rmsd) or (pdb_file, None, error_msg)
    """
    pdb_file, ref_ca_coords = args

    try:
        # Load mobile structure
        parser = PDB.PDBParser(QUIET=True)
        structure_id = os.path.basename(pdb_file).replace('.pdb', '')
        mobile_structure = parser.get_structure(structure_id, pdb_file)

        # Get mobile CA atoms
        mobile_ca_atoms = extract_ca_atoms(mobile_structure)
        mobile_ca_coords = np.array([atom.get_coord() for atom in mobile_ca_atoms])

        # Check if same number of CA atoms
        if len(mobile_ca_coords) != len(ref_ca_coords):
            return (pdb_file, None, f"CA atom count mismatch: {len(mobile_ca_coords)} vs {len(ref_ca_coords)}")

        # Kabsch algorithm for optimal superimposition
        # Center the coordinates
        ref_center = np.mean(ref_ca_coords, axis=0)
        mobile_center = np.mean(mobile_ca_coords, axis=0)

        ref_centered = ref_ca_coords - ref_center
        mobile_centered = mobile_ca_coords - mobile_center

        # Compute covariance matrix
        H = mobile_centered.T @ ref_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rotation and translation
        superimposed_coords = (mobile_ca_coords - mobile_center) @ R.T + ref_center

        # Calculate RMSD
        diff = superimposed_coords - ref_ca_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        return (pdb_file, superimposed_coords, rmsd)

    except Exception as e:
        return (pdb_file, None, str(e))


def run_pca_analysis(coords_matrix, pdb_files, output_dir, logger):
    """
    Run PCA analysis on superimposed coordinates.

    Args:
        coords_matrix: numpy array of shape (n_structures, n_atoms * 3)
        pdb_files: list of PDB file names
        output_dir: directory to save outputs
        logger: logging object
    """
    n_structures = coords_matrix.shape[0]
    n_components = min(10, n_structures - 1)  # Max 10 PCs or n-1

    logger.info(f"Running PCA with {n_components} components on {n_structures} structures")

    # Run PCA
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(coords_matrix)

    # Log explained variance
    logger.info("PCA Results:")
    logger.info("-" * 50)
    cumulative_var = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative_var += var
        logger.info(f"  PC{i+1}: {var*100:.2f}% variance (cumulative: {cumulative_var*100:.2f}%)")

    # Save PC scores to file
    pc_scores_file = os.path.join(output_dir, "pca_scores.txt")
    with open(pc_scores_file, 'w') as f:
        header = "PDB_file\t" + "\t".join([f"PC{i+1}" for i in range(n_components)])
        f.write(header + "\n")
        for i, pdb_file in enumerate(pdb_files):
            pdb_name = os.path.basename(pdb_file)
            scores = "\t".join([f"{s:.6f}" for s in pc_scores[i]])
            f.write(f"{pdb_name}\t{scores}\n")

    logger.info(f"PC scores saved to: {pc_scores_file}")

    # Save explained variance
    variance_file = os.path.join(output_dir, "pca_variance.txt")
    with open(variance_file, 'w') as f:
        f.write("PC\tExplained_Variance\tExplained_Variance_Ratio\tCumulative_Ratio\n")
        cumulative = 0
        for i in range(n_components):
            cumulative += pca.explained_variance_ratio_[i]
            f.write(f"PC{i+1}\t{pca.explained_variance_[i]:.6f}\t{pca.explained_variance_ratio_[i]:.6f}\t{cumulative:.6f}\n")

    logger.info(f"Variance explained saved to: {variance_file}")

    # Plot PC1 vs PC2 and scree plot
    if n_components >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PC1 vs PC2 scatter
        ax1 = axes[0]
        ax1.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.7, s=50)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax1.set_title('PCA: PC1 vs PC2')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Scree plot
        ax2 = axes[1]
        x_pos = range(1, n_components + 1)
        ax2.bar(x_pos, pca.explained_variance_ratio_ * 100, alpha=0.7, label='Individual')
        cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
        ax2.plot(x_pos, cumulative, 'ro-', label='Cumulative')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance (%)')
        ax2.set_title('Scree Plot')
        ax2.legend()
        ax2.set_xticks(x_pos)

        plt.tight_layout()
        plot_file = os.path.join(output_dir, "pca_plot.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()

        logger.info(f"PCA plot saved to: {plot_file}")

    # Plot PC1 vs PC3 and PC2 vs PC3 if available
    if n_components >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        ax1.scatter(pc_scores[:, 0], pc_scores[:, 2], alpha=0.7, s=50)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
        ax1.set_title('PCA: PC1 vs PC3')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        ax2 = axes[1]
        ax2.scatter(pc_scores[:, 1], pc_scores[:, 2], alpha=0.7, s=50)
        ax2.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
        ax2.set_title('PCA: PC2 vs PC3')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_file = os.path.join(output_dir, "pca_plot_pc3.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()

        logger.info(f"PC3 plots saved to: {plot_file}")

    return pca, pc_scores


def main():
    parser = argparse.ArgumentParser(
        description='Superimpose PDB structures by CA atoms and perform PCA analysis'
    )
    parser.add_argument('input_folder', help='Folder containing PDB files (searches recursively)')
    parser.add_argument('-o', '--output', help='Output directory (default: input_folder/pca_results)')
    parser.add_argument('-n', '--ncores', type=int, default=0,
                        help='Number of CPU cores to use (default: all available)')
    parser.add_argument('-r', '--reference', help='Reference PDB file (default: first PDB file)')

    args = parser.parse_args()

    # Setup paths
    input_folder = os.path.abspath(args.input_folder)
    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        output_dir = os.path.join(input_folder, 'pca_results')

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, 'pca_analysis.log')
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("PDB Structure PCA Analysis")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Find all PDB files recursively in all subdirectories
    pdb_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.pdb"), recursive=True))

    if len(pdb_files) < 2:
        logger.error(f"Need at least 2 PDB files for PCA, found {len(pdb_files)}")
        sys.exit(1)

    logger.info(f"Found {len(pdb_files)} PDB files")

    # Determine number of cores
    n_cores = args.ncores if args.ncores > 0 else cpu_count()
    n_cores = min(n_cores, len(pdb_files))
    logger.info(f"Using {n_cores} CPU cores for superimposition")

    # Load reference structure
    ref_pdb = args.reference if args.reference else pdb_files[0]
    logger.info(f"Reference structure: {os.path.basename(ref_pdb)}")

    ref_structure, _ = load_pdb(ref_pdb)
    if ref_structure is None:
        logger.error(f"Failed to load reference PDB: {ref_pdb}")
        sys.exit(1)

    ref_ca_coords = get_ca_coords(ref_structure)
    n_ca_atoms = len(ref_ca_coords)
    logger.info(f"Reference has {n_ca_atoms} CA atoms")

    # Superimpose all structures using multiprocessing
    logger.info("-" * 40)
    logger.info("Superimposing structures...")

    superimpose_args = [(pdb_file, ref_ca_coords) for pdb_file in pdb_files]

    with Pool(processes=n_cores) as pool:
        results = pool.map(superimpose_to_reference, superimpose_args)

    # Process results
    successful_files = []
    coords_list = []
    rmsd_values = []

    for pdb_file, coords, result in results:
        pdb_name = os.path.basename(pdb_file)
        if coords is not None:
            successful_files.append(pdb_file)
            coords_list.append(coords.flatten())  # Flatten to 1D for PCA
            rmsd_values.append(result)
            logger.info(f"  {pdb_name}: RMSD = {result:.3f} A")
        else:
            logger.warning(f"  {pdb_name}: FAILED - {result}")

    logger.info(f"Successfully superimposed {len(successful_files)}/{len(pdb_files)} structures")

    if len(successful_files) < 2:
        logger.error("Need at least 2 successfully superimposed structures for PCA")
        sys.exit(1)

    # Save RMSD values
    rmsd_file = os.path.join(output_dir, "rmsd_values.txt")
    with open(rmsd_file, 'w') as f:
        f.write("PDB_file\tRMSD_to_reference\n")
        for pdb_file, rmsd in zip(successful_files, rmsd_values):
            f.write(f"{os.path.basename(pdb_file)}\t{rmsd:.4f}\n")
    logger.info(f"RMSD values saved to: {rmsd_file}")

    # Create coordinate matrix for PCA
    coords_matrix = np.array(coords_list)
    logger.info(f"Coordinate matrix shape: {coords_matrix.shape}")

    # Run PCA
    logger.info("-" * 40)
    pca, pc_scores = run_pca_analysis(coords_matrix, successful_files, output_dir, logger)

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total PDB files: {len(pdb_files)}")
    logger.info(f"Successfully processed: {len(successful_files)}")
    logger.info(f"CA atoms per structure: {n_ca_atoms}")
    logger.info(f"Mean RMSD to reference: {np.mean(rmsd_values):.3f} A")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
