#!/usr/bin/env python3
import os
import argparse
import logging
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

def parse_args():
    parser = argparse.ArgumentParser(
        description='AF-ClaSEQ: Complete pipeline for sequence-based protein structure optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General options
    parser.add_argument('--input_a3m', required=True,
                        help='Path to input alignment file (A3M format)')
    parser.add_argument('--default_pdb', required=True,
                        help='Path to reference PDB structure')
    parser.add_argument('--base_dir', required=True,
                        help='Base directory for all output files')
    parser.add_argument('--config_path', required=True,
                        help='Path to configuration JSON file with filter criteria')
    parser.add_argument('--protein_name', required=True,
                        help='Name of the protein for job naming')
    
    # SLURM configuration
    parser.add_argument('--conda_env_path', required=True,
                        help='Path to conda environment')
    parser.add_argument('--slurm_account', required=True,
                        help='SLURM account for job submission')
    parser.add_argument('--slurm_partition', default='gpu',
                        help='SLURM partition for job submission')
    parser.add_argument('--slurm_time', default='04:00:00',
                        help='Wall time limit for SLURM jobs')
    parser.add_argument('--slurm_cpus_per_task', type=int, default=4,
                        help='Number of CPUs per task')
    parser.add_argument('--slurm_mem', default='16G',
                        help='Memory per job')
    parser.add_argument('--slurm_gpus', default='1',
                        help='Number of GPUs per job')
    parser.add_argument('--slurm_output', default='/dev/null',
                        help='Path for SLURM output')
    parser.add_argument('--slurm_error', default='/dev/null',
                        help='Path for SLURM error')
    parser.add_argument('--max_workers', type=int, default=64,
                        help='Maximum number of concurrent workers')
    
    # Pipeline control
    parser.add_argument('--stages', type=str, nargs='+', 
                        default=['01', '02', '03', '04', '05', '06'],
                        help='Pipeline stages to run (01-06)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # 01 - Iterative Shuffling parameters
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='Number of iterations for sequence shuffling')
    parser.add_argument('--num_shuffles', type=int, default=100,
                        help='Number of sequence shuffles per iteration')
    parser.add_argument('--seq_num_per_shuffle', type=int, default=128,
                        help='Number of sequences per shuffle')
    parser.add_argument('--plddt_threshold', type=float, default=70.0,
                        help='Minimum pLDDT score for structure filtering')
    
    # 02 - M-fold Sampling parameters
    parser.add_argument('--group_size', type=int, default=64,
                        help='Number of sequences per M-fold group')
    parser.add_argument('--random_select_num_seqs', type=int, default=None,
                        help='Number of sequences to randomly select (optional)')
    
    # 03 - Sequence Voting parameters
    parser.add_argument('--vote_threshold', type=int, default=1,
                        help='Minimum number of votes required')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of bins for metric quantization')
    parser.add_argument('--min_value', type=float, default=None,
                        help='Minimum value for metric binning')
    parser.add_argument('--max_value', type=float, default=None,
                        help='Maximum value for metric binning')
    
    # 04 - Recompilation & Prediction parameters
    parser.add_argument('--bin_numbers', type=int, nargs='+',
                        help='Bin numbers to select for final prediction')
    parser.add_argument('--combine_bins', action='store_true',
                        help='Combine sequences from all selected bins')
    parser.add_argument('--num_selections', type=int, default=10,
                        help='Number of random selections to create for comparison')
    parser.add_argument('--seq_num_per_selection', type=int, default=128,
                        help='Number of sequences per random selection')
    
    # Additional options
    parser.add_argument('--check_interval', type=int, default=60,
                        help='Interval (seconds) to check job status')
    
    return parser.parse_args()

def setup_logging(base_dir: str):
    """Set up logging configuration"""
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "af_claseq_pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def run_command(cmd: List[str], description: str) -> int:
    """Run a command with proper logging"""
    cmd_str = " ".join(cmd)
    logging.info(f"Running {description}: {cmd_str}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}")
        logging.error(f"STDOUT: {result.stdout}")
        logging.error(f"STDERR: {result.stderr}")
    else:
        logging.info(f"Command completed successfully")
        
    return result.returncode

def stage_01_iterative_shuffling(args):
    """Run the iterative shuffling stage"""
    logging.info("=== STAGE 01: ITERATIVE SHUFFLING ===")
    
    cmd = [
        "python", "scripts/01_iterative_shuffling_run.py",
        "--input_a3m", args.input_a3m,
        "--default_pdb", args.default_pdb,
        "--base_dir", args.base_dir,
        "--config_path", args.config_path,
        "--num_iterations", str(args.num_iterations),
        "--num_shuffles", str(args.num_shuffles),
        "--seq_num_per_shuffle", str(args.seq_num_per_shuffle),
        "--plddt_threshold", str(args.plddt_threshold),
        "--conda_env_path", args.conda_env_path,
        "--slurm_account", args.slurm_account,
        "--slurm_partition", args.slurm_partition,
        "--slurm_cpus_per_task", str(args.slurm_cpus_per_task),
        "--slurm_mem", args.slurm_mem,
        "--slurm_time", args.slurm_time,
        "--slurm_output", args.slurm_output,
        "--slurm_error", args.slurm_error,
        "--job_name_prefix", f"{args.protein_name}_shuffle",
        "--random_seed", str(args.random_seed),
        "--check_interval", str(args.check_interval),
        "--max_workers", str(args.max_workers)
    ]
    
    if run_command(cmd, "iterative shuffling") != 0:
        logging.error("Stage 01 failed")
        return False
    
    # Also run the plotting script for visualization
    plot_cmd = [
        "python", "scripts/01_iteration_shuffling_plot.py",
        "--parent_dir", args.base_dir,
        "--config_path", args.config_path
    ]
    run_command(plot_cmd, "iteration shuffling plot generation")
    
    # Run the A3M combine script to collect filtered sequences
    combine_cmd = [
        "python", "scripts/01_iteration_shuffling_a3m-combine.py",
        "--parent_dir", args.base_dir,
        "--config_path", args.config_path,
        "--default_pdb", args.default_pdb,
        "--max_workers", str(args.max_workers),
        "--threshold", "0.5"  # Default threshold, might want to make configurable
    ]
    run_command(combine_cmd, "iteration shuffling sequence combination")
    
    return True

def stage_02_m_fold_sampling(args):
    """Run the M-fold sampling stage"""
    logging.info("=== STAGE 02: M-FOLD SAMPLING ===")
    
    # Determine input MSA for this stage
    input_a3m = os.path.join(args.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")
    if not os.path.exists(input_a3m):
        logging.warning(f"Expected output from stage 01 not found: {input_a3m}")
        logging.warning("Using original input MSA instead")
        input_a3m = args.input_a3m
    
    cmd = [
        "python", "scripts/02_M_fold_sampling_run.py",
        "--input_a3m", input_a3m,
        "--default_pdb", args.default_pdb,
        "--base_dir", args.base_dir,
        "--group_size", str(args.group_size),
        "--conda_env_path", args.conda_env_path,
        "--slurm_account", args.slurm_account,
        "--slurm_partition", args.slurm_partition,
        "--slurm_gpus", args.slurm_gpus,
        "--slurm_cpus_per_task", str(args.slurm_cpus_per_task),
        "--slurm_mem", args.slurm_mem,
        "--slurm_time", args.slurm_time,
        "--slurm_output", args.slurm_output,
        "--slurm_error", args.slurm_error,
        "--job_name_prefix", f"{args.protein_name}_mfold",
        "--check_interval", str(args.check_interval),
        "--max_workers", str(args.max_workers)
    ]
    
    if args.random_select_num_seqs:
        cmd.extend(["--random_select_num_seqs", str(args.random_select_num_seqs)])
    
    if run_command(cmd, "M-fold sampling") != 0:
        logging.error("Stage 02 failed")
        return False
    
    # Generate 1D plots
    plot_1d_cmd = [
        "python", "scripts/02_sampling_plot_1D_CV.py",
        "--results_dir", os.path.join(args.base_dir, "02_sampling"),
        "--output_dir", os.path.join(args.base_dir, "02_sampling/plots/1D"),
        "--csv_dir", os.path.join(args.base_dir, "02_sampling/csv"),
        "--config_path", args.config_path
    ]
    run_command(plot_1d_cmd, "1D sampling plot generation")
    
    # Generate 2D plots
    plot_2d_cmd = [
        "python", "scripts/02_sampling_plot_2D_CV.py",
        "--results_dir", os.path.join(args.base_dir, "02_sampling"),
        "--output_dir", os.path.join(args.base_dir, "02_sampling/plots/2D"),
        "--csv_dir", os.path.join(args.base_dir, "02_sampling/csv"),
        "--config_path", args.config_path
    ]
    run_command(plot_2d_cmd, "2D sampling plot generation")
    
    return True

def stage_03_sequence_voting(args):
    """Run the sequence voting stage"""
    logging.info("=== STAGE 03: SEQUENCE VOTING ===")
    
    # Determine input MSA for this stage (either original or from stage 01)
    source_msa = args.input_a3m
    if os.path.exists(os.path.join(args.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")):
        source_msa = os.path.join(args.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")
    
    cmd = [
        "python", "scripts/03_sequence_voting_run.py",
        "--sampling_dir", os.path.join(args.base_dir, "02_sampling"),
        "--source_msa", source_msa,
        "--config_path", args.config_path,
        "--output_dir", os.path.join(args.base_dir, "03_voting"),
        "--num_bins", str(args.num_bins),
        "--vote_threshold", str(args.vote_threshold),
        "--plddt_threshold", str(args.plddt_threshold),
        "--max_workers", str(args.max_workers)
    ]
    
    if args.min_value is not None:
        cmd.extend(["--min_value", str(args.min_value)])
    
    if args.max_value is not None:
        cmd.extend(["--max_value", str(args.max_value)])
    
    if run_command(cmd, "sequence voting") != 0:
        logging.error("Stage 03 failed")
        return False
    
    return True

def stage_04_recompile_and_predict(args):
    """Run the recompilation and prediction stage"""
    logging.info("=== STAGE 04: SEQUENCE RECOMPILATION & PREDICTION ===")
    
    # If bin_numbers not specified, try to determine best bins from voting results
    if not args.bin_numbers:
        # This would require analyzing the voting results
        # For simplicity, we'll use a default range of higher bins
        logging.warning("No bin numbers specified, using default range (5-10)")
        bin_numbers = list(range(5, 11))
    else:
        bin_numbers = args.bin_numbers
    
    # Recompile sequences
    source_msa = args.input_a3m
    if os.path.exists(os.path.join(args.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")):
        source_msa = os.path.join(args.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")
    
    recompile_cmd = [
        "python", "scripts/04_recompile_run.py",
        "--output_dir", os.path.join(args.base_dir, "04_recompile"),
        "--bin_numbers", *map(str, bin_numbers),
        "--source_msa", source_msa,
        "--default_pdb", args.default_pdb,
        "--voting_results", os.path.join(args.base_dir, "03_voting/03_voting_results.csv"),
        "--num_selections", str(args.num_selections),
        "--seq_num_per_selection", str(args.seq_num_per_selection)
    ]
    
    if args.combine_bins:
        recompile_cmd.append("--combine_bins")
    
    if run_command(recompile_cmd, "sequence recompilation") != 0:
        logging.error("Recompilation stage failed")
        return False
    
    # Run predictions
    predict_cmd = [
        "python", "scripts/04_prediction_run.py",
        "--base_dir", os.path.join(args.base_dir, "04_recompile"),
        "--bin_numbers", *map(str, bin_numbers),
        "--slurm_account", args.slurm_account,
        "--slurm_partition", args.slurm_partition, 
        "--slurm_output", args.slurm_output,
        "--slurm_error", args.slurm_error,
        "--conda_env_path", args.conda_env_path,
        "--slurm_time", args.slurm_time,
        "--slurm_cpus_per_task", str(args.slurm_cpus_per_task),
        "--slurm_mem", args.slurm_mem,
        "--slurm_gpus", args.slurm_gpus,
        "--check_interval", str(args.check_interval),
        "--max_workers", str(args.max_workers)
    ]
    
    if args.combine_bins:
        predict_cmd.append("--combine_bins")
    
    if run_command(predict_cmd, "structure prediction") != 0:
        logging.error("Prediction stage failed")
        return False
    
    return True

def stage_05_plot_results(args):
    """Run the result plotting stage"""
    logging.info("=== STAGE 05: RESULT ANALYSIS & PLOTTING ===")
    
    plot_cmd = [
        "python", "scripts/05_plot.py",
        "--base_dir", os.path.join(args.base_dir, "04_recompile"),
        "--output_dir", os.path.join(args.base_dir, "05_analysis"),
        "--config_path", args.config_path
    ]
    
    if run_command(plot_cmd, "result plotting") != 0:
        logging.error("Stage 05 failed")
        return False
    
    # Also run local pLDDT analysis
    plddt_cmd = [
        "python", "scripts/local_plddt.py",
        "--base_dir", os.path.join(args.base_dir, "04_recompile"),
        "--output_dir", os.path.join(args.base_dir, "05_analysis/local_plddt"),
        "--config", args.config_path,
        "--window_size", "5"  # Default window size
    ]
    run_command(plddt_cmd, "local pLDDT analysis")
    
    return True

def stage_06_phylogenetic_analysis(args):
    """Run the phylogenetic analysis stage"""
    logging.info("=== STAGE 06: PHYLOGENETIC ANALYSIS ===")
    
    # This stage would require identifying the appropriate A3M files for visualization
    # As a simplification, we'll use files from stage 04
    tree_cmd = [
        "python", "scripts/06_tree.py",
        "--newick_tree", os.path.join(args.base_dir, "tree.nwk"),  # This would need to be created separately
        "--output_dir", os.path.join(args.base_dir, "06_phylogeny"),
    ]
    
    # Add state A3Ms, colors, and names based on bin selections
    if args.bin_numbers:
        for bin_num in args.bin_numbers:
            a3m_path = os.path.join(args.base_dir, f"04_recompile/prediction/bin_{bin_num}/bin{bin_num}_sequences.a3m")
            if os.path.exists(a3m_path):
                tree_cmd.extend([
                    "--state_a3ms", a3m_path,
                    "--state_colors", "#468F8B",  # Default color from the code
                    "--state_names", f"Bin {bin_num}"
                ])
    
    # Also add control sequences
    control_a3m = os.path.join(args.base_dir, "04_recompile/control_prediction/bin_control/control_sequences.a3m")
    if os.path.exists(control_a3m):
        tree_cmd.extend([
            "--state_a3ms", control_a3m,
            "--state_colors", "#AFD2D0",  # Default control color from the code
            "--state_names", "Control"
        ])
    
    # Check if we have A3Ms to visualize
    if "--state_a3ms" not in tree_cmd:
        logging.warning("No A3M files found for phylogenetic analysis, skipping stage 06")
        return False
    
    if not os.path.exists(os.path.join(args.base_dir, "tree.nwk")):
        logging.warning("Newick tree file not found, skipping stage 06")
        return False
    
    if run_command(tree_cmd, "phylogenetic analysis") != 0:
        logging.error("Stage 06 failed")
        return False
    
    return True

def main():
    args = parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.base_dir)
    logging.info("=== AF-ClaSEQ PIPELINE STARTED ===")
    logging.info("Arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    
    # Create dirs for each stage
    for stage in ["01_iterative_shuffling", "02_sampling", "03_voting", 
                 "04_recompile", "05_analysis", "06_phylogeny"]:
        os.makedirs(os.path.join(args.base_dir, stage), exist_ok=True)
    
    # Run selected stages
    stages_to_run = args.stages
    
    if "01" in stages_to_run:
        if not stage_01_iterative_shuffling(args):
            logging.error("Stopping pipeline due to failure in stage 01")
            return
    
    if "02" in stages_to_run:
        if not stage_02_m_fold_sampling(args):
            logging.error("Stopping pipeline due to failure in stage 02")
            return
    
    if "03" in stages_to_run:
        if not stage_03_sequence_voting(args):
            logging.error("Stopping pipeline due to failure in stage 03")
            return
    
    if "04" in stages_to_run:
        if not stage_04_recompile_and_predict(args):
            logging.error("Stopping pipeline due to failure in stage 04")
            return
    
    if "05" in stages_to_run:
        if not stage_05_plot_results(args):
            logging.error("Stopping pipeline due to failure in stage 05")
            return
    
    if "06" in stages_to_run:
        if not stage_06_phylogenetic_analysis(args):
            logging.warning("Stage 06 (phylogenetic analysis) was not completed")
    
    logging.info("=== AF-ClaSEQ PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()