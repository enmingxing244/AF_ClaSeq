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
        description='Complete pipeline of AF-ClaSeq including all the stages',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # =====================================================================
    # General options
    # =====================================================================
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument('--source_a3m', required=True,
                        help='Path to raw MSA alignment file (A3M format)')
    general_group.add_argument('--default_pdb', required=True,
                        help='Path to ColabFold default predicted PDB structure')
    general_group.add_argument('--base_dir', required=True,
                        help='Base directory for all output files')
    general_group.add_argument('--config_file', required=True,
                        help='Path to configuration JSON file with filter criteria')
    general_group.add_argument('--protein_name', required=True,
                        help='Name of the protein for job naming')
    general_group.add_argument('--coverage_threshold', type=float, default=0.8,
                        help='Minimum required sequence coverage (default: 0.8)')
    general_group.add_argument('--num_models', type=int, default=1,
                        help='Number of models to predict per structure')
    general_group.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    general_group.add_argument('--num_bins', type=int, default=10,
                        help='Number of bins for metric quantization (1-100)')
    general_group.add_argument('--plot_initial_color', type=str, default='#87CEEB',
                        help='Initial color in hex format for gradient in plots (e.g. #d5f6dd, default: #87CEEB skyblue)')
    general_group.add_argument('--plot_end_color', type=str, default='#FFFFFF',
                        help='End color in hex format for gradient in plots (e.g. #d5f6dd, default: #FFFFFF white)')
 
    
    # =====================================================================
    # SLURM configuration
    # =====================================================================
    slurm_group = parser.add_argument_group('SLURM Configuration')
    slurm_group.add_argument('--conda_env_path', default='/fs/ess/PAA0203/xing244/.conda/envs/colabfold',
                        help='Path to conda environment')
    slurm_group.add_argument('--slurm_account', default='PAA0203',
                        help='SLURM account for job submission')
    slurm_group.add_argument('--slurm_output', default='/dev/null',
                        help='Path for SLURM standard output logs')
    slurm_group.add_argument('--slurm_error', default='/dev/null',
                        help='Path for SLURM error logs')
    slurm_group.add_argument('--slurm_nodes', type=int, default=1,
                        help='Number of nodes per SLURM job')
    slurm_group.add_argument('--slurm_gpus_per_task', type=int, default=1,
                        help='Number of GPUs per task')
    slurm_group.add_argument('--slurm_tasks', type=int, default=1,
                        help='Number of tasks per SLURM job')
    slurm_group.add_argument('--slurm_cpus_per_task', type=int, default=4,
                        help='Number of CPUs per task')
    slurm_group.add_argument('--slurm_time', default='04:00:00',
                        help='Wall time limit for SLURM jobs in format HH:MM:SS')  
    slurm_group.add_argument('--slurm_partition', default='nextgen',
                        help='SLURM partition for job submission (e.g., gpu, cpu)')
    slurm_group.add_argument('--max_workers', type=int, default=64,
                        help='Maximum number of concurrent workers/jobs')
    
    # =====================================================================
    # Pipeline control
    # =====================================================================
    pipeline_group = parser.add_argument_group('Pipeline Control')
    pipeline_group.add_argument('--stages', type=str, nargs='+', 
                        default=['01_RUN', '01_ANALYSIS', '02_RUN', '02_ANALYSIS', '03', '04', '05', '06'],
                        help='Pipeline stages to run (01_RUN=Iterative Shuffling, 01_ANALYSIS=Iterative Shuffling Analysis, '
                             '02_RUN=M-fold Sampling, 02_ANALYSIS=M-fold Sampling Analysis, '
                             '03=Sequence Voting, 04=Recompilation, 05=Result Analysis, 06=Phylogeny)')
    pipeline_group.add_argument('--check_interval', type=int, default=60,
                        help='Interval (seconds) to check job status')
    
    # =====================================================================
    # 01 - Iterative Shuffling enrichment parameters
    # =====================================================================
    shuffle_group = parser.add_argument_group('Stage 01: Iterative Shuffling Parameters')
    shuffle_group.add_argument('--iter_shuf_input_a3m', required=True,
                        help='Path to input alignment file (A3M format) for iterative shuffling')
    shuffle_group.add_argument('--num_iterations', type=int, default=8,
                        help='Number of iterations for sequence shuffling')
    shuffle_group.add_argument('--num_shuffles', type=int, default=10,
                        help='Number of sequence shuffles per iteration')
    shuffle_group.add_argument('--seq_num_per_shuffle', type=int, default=16,
                        help='Number of sequences per shuffle')
    shuffle_group.add_argument('--plddt_threshold', type=int, default=75,
                        help='Minimum pLDDT score for structure filtering (0-100)')
    shuffle_group.add_argument('--quantile', type=float, default=0.2,
                        help='Quantile threshold for filtering')
    shuffle_group.add_argument('--resume_from_iter', type=int,
                        help='Resume from a specific iteration number\n'
                        '(e.g., 2 to resume from iteration 2 using iteration 1 results)')
    
    shuffle_group.add_argument('--iter_shuf_plot_num_cols', type=int, default=5, 
                        help='Number of columns in plot grid')
    shuffle_group.add_argument('--iter_shuf_plot_x_min', type=float, default=0, 
                        help='Minimum x-axis value')
    shuffle_group.add_argument('--iter_shuf_plot_x_max', type=float, default=20, 
                        help='Maximum x-axis value')
    shuffle_group.add_argument('--iter_shuf_plot_y_min', type=float, default=0.8, 
                        help='Minimum y-axis value')
    shuffle_group.add_argument('--iter_shuf_plot_y_max', type=float, default=10000, 
                        help='Maximum y-axis value')
    shuffle_group.add_argument('--iter_shuf_plot_xticks', type=float, nargs='+', 
                        help='List of x-axis tick positions')
    shuffle_group.add_argument('--iter_shuf_plot_bin_step', type=float, default=0.2, 
                        help='Step size for binning')
    
    shuffle_group.add_argument('--iter_shuf_combine_threshold', type=float, required=True,
                        help='Threshold for filtering (TM-score > threshold or RMSD < threshold)')
    
    # =====================================================================
    # 02 - M-fold Sampling parameters
    # =====================================================================
    mfold_group = parser.add_argument_group('Stage 02: M-fold Sampling Parameters')
    mfold_group.add_argument('--m_fold_samp_input_a3m', required=True,
                        help='Path to input alignment file (A3M format) for M-fold sampling')
    mfold_group.add_argument('--group_size', type=int, default=64,
                        help='Number of sequences per M-fold group')
    mfold_group.add_argument('--random_select_num_seqs', type=int, default=None,
                        help='Number of sequences to randomly select from input MSA \n'
                             'to reduce the size of MSA for M-fold sampling (optional)')
    
    # M-fold 1D sampling plot parameters
    mfold_group.add_argument('--mfold_1d_plot_x_min', type=float, default=None,
                        help='Minimum x-axis value for M-fold 1D sampling plots')
    mfold_group.add_argument('--mfold_1d_plot_x_max', type=float, default=None,
                        help='Maximum x-axis value for M-fold 1D sampling plots')
    mfold_group.add_argument('--mfold_1d_plot_log_scale', action='store_true',
                        help='Use log scale for y-axis in M-fold 1D sampling distribution plots')
    mfold_group.add_argument('--mfold_1d_plot_gradient_ascending', action='store_true',
                        help='Use ascending gradient for color in M-fold 1D sampling plots')
    mfold_group.add_argument('--mfold_1d_plot_linear_gradient', action='store_true',
                        help='Use linear gradient for color in M-fold 1D sampling plots')
    mfold_group.add_argument('--mfold_1d_plot_plddt_threshold', type=float, default=0,
                        help='pLDDT threshold for filtering structures in 1D plots (default: 0, no filtering)')
    mfold_group.add_argument('--mfold_1d_plot_figsize', type=float, nargs=2, default=(10, 5),
                        help='Figure size in inches (width, height) for M-fold 1D sampling plots (default: 10 5)')
    mfold_group.add_argument('--mfold_1d_plot_show_bin_lines', action='store_true', default=False,
                        help='Show vertical dashed lines at bin boundaries in M-fold 1D sampling plots')
    mfold_group.add_argument('--mfold_1d_plot_y_min', type=float, default=None,
                        help='Minimum y-axis value for M-fold 1D sampling plots')
    mfold_group.add_argument('--mfold_1d_plot_y_max', type=float, default=None,
                        help='Maximum y-axis value for M-fold 1D sampling plots')
    mfold_group.add_argument('--mfold_1d_plot_x_ticks', nargs='*', type=float, default=None,
                        help='List of x-axis tick values for M-fold 1D sampling plots (optional)')
    
    # M-fold 2D sampling plot parameters
    mfold_group.add_argument('--mfold_2d_plot_x_min', type=float, default=None,
                        help='Minimum x-axis value for M-fold 2D sampling plots')
    mfold_group.add_argument('--mfold_2d_plot_x_max', type=float, default=None,
                        help='Maximum x-axis value for M-fold 2D sampling plots')
    mfold_group.add_argument('--mfold_2d_plot_y_min', type=float, default=None,
                        help='Minimum y-axis value for M-fold 2D sampling plots')
    mfold_group.add_argument('--mfold_2d_plot_y_max', type=float, default=None,
                        help='Maximum y-axis value for M-fold 2D sampling plots')
    mfold_group.add_argument('--mfold_2d_plot_plddt_threshold', type=float, default=0,
                        help='pLDDT threshold for filtering structures in 2D plots (default: 0, no filtering)')
    mfold_group.add_argument('--mfold_2d_plot_x_ticks', nargs='*', type=float, default=None,
                        help='List of x-axis tick values for M-fold 2D sampling plots')
    mfold_group.add_argument('--mfold_2d_plot_y_ticks', nargs='*', type=float, default=None,
                        help='List of y-axis tick values for M-fold 2D sampling plots')
    # =====================================================================
    # 03 - Sequence Voting parameters
    # =====================================================================
    voting_group = parser.add_argument_group('Stage 03: Sequence Voting Parameters')
    voting_group.add_argument('--vote_threshold', type=int, default=1,
                        help='Minimum number of votes required for a sequence to be selected')
    voting_group.add_argument('--vote_min_value', type=float, default=None,
                        help='Minimum value for metric binning (optional, auto-determined if not specified)')
    voting_group.add_argument('--vote_max_value', type=float, default=None,
                        help='Maximum value for metric binning (optional, auto-determined if not specified)')
    voting_group.add_argument('--vote_figsize', type=float, nargs=2, default=(10, 5),
                        help='Figure size in inches (width, height) for voting distribution plots (default: 10 5)')
    voting_group.add_argument('--vote_y_min', type=float, default=None,
                        help='Minimum y-axis value for voting distribution plot')
    voting_group.add_argument('--vote_y_max', type=float, default=None,
                        help='Maximum y-axis value for voting distribution plot')
    voting_group.add_argument('--vote_x_ticks', type=int, nargs='+', default=None,
                        help='Custom x-axis tick positions for voting distribution plot')
    voting_group.add_argument('--vote_hierarchical_sampling', action='store_true',
                        help='Use hierarchical sampling directories structure for voting analysis')
    
    # =====================================================================
    # 04 - Recompilation & Prediction parameters
    # =====================================================================
    recompile_group = parser.add_argument_group('Stage 04: Recompilation & Prediction Parameters')
    recompile_group.add_argument('--bin_numbers', type=int, nargs='+',
                        help='Bin numbers to select for final prediction (e.g., 8 9 10 for top 3 bins)')
    recompile_group.add_argument('--combine_bins', action='store_true',
                        help='Combine sequences from all selected bins into a single prediction')
    recompile_group.add_argument('--num_total_bins', type=int, default=30,
                        help='Total number of bins used in voting')
    recompile_group.add_argument('--prediction_num_model', type=int, default=5,
                        help='Number of models to use for prediction')
    recompile_group.add_argument('--prediction_num_seed', type=int, default=8,
                        help='Number of seeds to use for prediction')

    
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

def stage_01_run_iterative_shuffling(args):
    """Run the iterative shuffling stage"""
    logging.info("=== STAGE 01_RUN: ITERATIVE SHUFFLING ===")
    
    cmd = [
        "python", "scripts/01_iterative_shuffling_run.py",
        "--iter_shuf_input_a3m", args.iter_shuf_input_a3m,
        "--default_pdb", args.default_pdb,
        "--base_dir", args.base_dir,
        "--seq_num_per_shuffle", str(args.seq_num_per_shuffle),
        "--num_shuffles", str(args.num_shuffles),
        "--coverage_threshold", str(args.coverage_threshold),
        "--num_iterations", str(args.num_iterations),
        "--quantile", str(args.quantile),
        "--config_file", args.config_file,
        "--plddt_threshold", str(args.plddt_threshold),
        "--num_models", str(args.num_models),
        
        "--conda_env_path", args.conda_env_path,
        "--slurm_account", args.slurm_account,
        "--slurm_output", args.slurm_output,
        "--slurm_error", args.slurm_error,
        "--slurm_nodes", str(args.slurm_nodes),
        "--slurm_gpus_per_task", str(args.slurm_gpus_per_task),
        "--slurm_tasks", str(args.slurm_tasks),
        "--slurm_cpus_per_task", str(args.slurm_cpus_per_task),
        "--slurm_time", args.slurm_time,
        "--slurm_partition", args.slurm_partition,
        "--random_seed", str(args.random_seed),
        "--check_interval", str(args.check_interval),
        "--max_workers", str(args.max_workers)
    ]
    
    # Add optional resume_from_iter if specified
    if args.resume_from_iter is not None:
        cmd.extend(["--resume_from_iter", str(args.resume_from_iter)])
    
    if run_command(cmd, "iterative shuffling") != 0:
        logging.error("Stage 01_RUN failed")
        return False
    
    return True

def stage_01_analysis_iterative_shuffling(args):
    """Run the iterative shuffling analysis stage"""
    logging.info("=== STAGE 01_ANALYSIS: ITERATIVE SHUFFLING ANALYSIS ===")
    
    # Run the plotting script for visualization
    plot_cmd = [
        "python", "scripts/01_iteration_shuffling_plot.py",
        "--base_dir", args.base_dir,
        "--config_path", args.config_file,
        "--iter_shuf_plot_num_cols", str(args.iter_shuf_plot_num_cols),
        "--iter_shuf_plot_x_min", str(args.iter_shuf_plot_x_min),
        "--iter_shuf_plot_x_max", str(args.iter_shuf_plot_x_max),
        "--iter_shuf_plot_y_min", str(args.iter_shuf_plot_y_min),
        "--iter_shuf_plot_y_max", str(args.iter_shuf_plot_y_max),
        "--iter_shuf_plot_bin_step", str(args.iter_shuf_plot_bin_step)
    ]
    
    # Add optional xticks if specified
    if args.iter_shuf_plot_xticks is not None:
        xticks_str = " ".join(str(x) for x in args.iter_shuf_plot_xticks)
        plot_cmd.extend(["--iter_shuf_plot_xticks", *xticks_str.split()])
    
    if run_command(plot_cmd, "iteration shuffling plot generation") != 0:
        logging.error("Iteration shuffling plot generation failed")
        return False
    
    # Run the A3M combine script to collect filtered sequences
    combine_cmd = [
        "python", "scripts/01_iteration_shuffling_a3m-combine.py",
        "--base_dir", args.base_dir,
        "--config_path", args.config_file,
        "--default_pdb", args.default_pdb,
        "--max_workers", str(args.max_workers),
        "--iter_shuf_combine_threshold", str(args.iter_shuf_combine_threshold)
    ]
    
    if run_command(combine_cmd, "iteration shuffling sequence combination") != 0:
        logging.error("Iteration shuffling sequence combination failed")
        return False
    
    return True

def stage_02_run_m_fold_sampling(args):
    """Run the M-fold sampling stage"""
    logging.info("=== STAGE 02_RUN: M-FOLD SAMPLING ===")
    
    # Determine input MSA for this stage
    m_fold_samp_input_a3m = os.path.join(args.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")
    if not os.path.exists(m_fold_samp_input_a3m):
        logging.warning(f"Expected output from stage 01 not found: {m_fold_samp_input_a3m}")
        logging.warning("Using original input MSA instead")
        m_fold_samp_input_a3m = args.m_fold_samp_input_a3m
    
    cmd = [
        "python", "scripts/02_M_fold_sampling_run.py",
        "--m_fold_samp_input_a3m", m_fold_samp_input_a3m,
        "--default_pdb", args.default_pdb,
        "--base_dir", args.base_dir,
        "--group_size", str(args.group_size),
        "--coverage_threshold", str(args.coverage_threshold),
        "--num_models", str(args.num_models),
        
        "--conda_env_path", args.conda_env_path,
        "--slurm_account", args.slurm_account,
        "--slurm_output", args.slurm_output,
        "--slurm_error", args.slurm_error,
        "--slurm_nodes", str(args.slurm_nodes),
        "--slurm_gpus_per_task", str(args.slurm_gpus_per_task),
        "--slurm_tasks", str(args.slurm_tasks),
        "--slurm_cpus_per_task", str(args.slurm_cpus_per_task),
        "--slurm_time", args.slurm_time,
        "--slurm_partition", args.slurm_partition,
        "--random_seed", str(args.random_seed),
        "--check_interval", str(args.check_interval),
        "--max_workers", str(args.max_workers)
    ]
    
    if args.random_select_num_seqs:
        cmd.extend(["--random_select_num_seqs", str(args.random_select_num_seqs)])
    
    if run_command(cmd, "M-fold sampling") != 0:
        logging.error("Stage 02_RUN failed")
        return False
    
    return True
def stage_02_analysis_m_fold_sampling(args):
    """Run the M-fold sampling analysis stage"""
    logging.info("=== STAGE 02_ANALYSIS: M-FOLD SAMPLING ANALYSIS ===")
    
    # Generate 1D plots
    plot_1d_cmd = [
        "python", "scripts/02_sampling_plot_1D_CV.py",
        "--results_dir", os.path.join(args.base_dir, "02_sampling"),
        "--output_dir", os.path.join(args.base_dir, "02_sampling_plot/1D"),
        "--csv_dir", os.path.join(args.base_dir, "02_sampling_plot/csv"),
        "--config_file", args.config_file,
        "--initial_color", args.plot_initial_color,
        "--end_color", args.plot_end_color,
        "--n_plot_bins", str(args.num_bins),
        "--plddt_threshold", str(args.mfold_1d_plot_plddt_threshold)
    ]
    
    # Add optional parameters if they are set
    if args.mfold_1d_plot_log_scale:
        plot_1d_cmd.append("--log_scale")
    
    if args.mfold_1d_plot_gradient_ascending:
        plot_1d_cmd.append("--gradient_ascending")
    
    if args.mfold_1d_plot_linear_gradient:
        plot_1d_cmd.append("--linear_gradient")
    
    if args.mfold_1d_plot_show_bin_lines:
        plot_1d_cmd.append("--show_bin_lines")
    
    if args.mfold_1d_plot_x_min is not None:
        plot_1d_cmd.extend(["--x_min", str(args.mfold_1d_plot_x_min)])
    
    if args.mfold_1d_plot_x_max is not None:
        plot_1d_cmd.extend(["--x_max", str(args.mfold_1d_plot_x_max)])
    
    if args.mfold_1d_plot_y_min is not None:
        plot_1d_cmd.extend(["--y_min", str(args.mfold_1d_plot_y_min)])
    
    if args.mfold_1d_plot_y_max is not None:
        plot_1d_cmd.extend(["--y_max", str(args.mfold_1d_plot_y_max)])
    
    if args.mfold_1d_plot_figsize:
        plot_1d_cmd.extend(["--figsize", str(args.mfold_1d_plot_figsize[0]), str(args.mfold_1d_plot_figsize[1])])
    
    if args.mfold_1d_plot_x_ticks:
        plot_1d_cmd.extend(["--x_ticks"] + [str(tick) for tick in args.mfold_1d_plot_x_ticks])
    
    
    if run_command(plot_1d_cmd, "1D sampling plot generation") != 0:
        logging.error("1D sampling plot generation failed")
        return False
    
    # Generate 2D plots
    plot_2d_cmd = [
        "python", "scripts/02_sampling_plot_2D_CV.py",
        "--results_dir", os.path.join(args.base_dir, "02_sampling"),
        "--output_dir", os.path.join(args.base_dir, "02_sampling_plot/2D"),
        "--csv_dir", os.path.join(args.base_dir, "02_sampling_plot/csv"),
        "--config_file", args.config_file,
        "--plddt_threshold", str(args.mfold_2d_plot_plddt_threshold)
    ]
    
    # Add optional parameters if they are set
    if args.mfold_2d_plot_x_min is not None:
        plot_2d_cmd.extend(["--x_min", str(args.mfold_2d_plot_x_min)])
    
    if args.mfold_2d_plot_x_max is not None:
        plot_2d_cmd.extend(["--x_max", str(args.mfold_2d_plot_x_max)])
    
    if args.mfold_2d_plot_y_min is not None:
        plot_2d_cmd.extend(["--y_min", str(args.mfold_2d_plot_y_min)])
    
    if args.mfold_2d_plot_y_max is not None:
        plot_2d_cmd.extend(["--y_max", str(args.mfold_2d_plot_y_max)])
    
    if args.mfold_2d_plot_x_ticks:
        plot_2d_cmd.extend(["--x_ticks"] + [str(tick) for tick in args.mfold_2d_plot_x_ticks])
    
    if args.mfold_2d_plot_y_ticks:
        plot_2d_cmd.extend(["--y_ticks"] + [str(tick) for tick in args.mfold_2d_plot_y_ticks])
    
    if run_command(plot_2d_cmd, "2D sampling plot generation") != 0:
        logging.error("2D sampling plot generation failed")
        return False
    
    return True

def stage_03_sequence_voting(args):
    """Run the sequence voting stage"""
    logging.info("=== STAGE 03: SEQUENCE VOTING ===")
    
    # Determine input MSA for this stage (either original or from stage 01)
    source_msa = args.source_a3m
    if os.path.exists(os.path.join(args.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")):
        source_msa = os.path.join(args.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")
    
    cmd = [
        "python", "scripts/03_sequence_voting_run.py",
        "--sampling_dir", os.path.join(args.base_dir, "02_sampling"),
        "--source_msa", source_msa,
        "--config_path", args.config_file,
        "--output_dir", os.path.join(args.base_dir, "03_voting"),
        "--num_bins", str(args.num_bins),
        "--vote_threshold", str(args.vote_threshold),
        "--plddt_threshold", str(args.plddt_threshold),
        "--max_workers", str(args.max_workers),
        "--initial_color", args.plot_initial_color,
        "--end_color", args.vote_end_color,
        "--use_focused_bins", str(args.use_focused_bins),
        "--precomputed_metrics", os.path.join(args.base_dir, "02_sampling/csv/*.csv"),
        "--figsize", str(args.vote_figsize[0]), str(args.vote_figsize[1])
    ]
    
    if args.min_value is not None:
        cmd.extend(["--vote_min_value", str(args.vote_min_value)])
    
    if args.vote_max_value is not None:
        cmd.extend(["--vote_max_value", str(args.vote_max_value)])
    
    if args.vote_y_min is not None:
        cmd.extend(["--y_min", str(args.vote_y_min)])
    
    if args.vote_y_max is not None:
        cmd.extend(["--y_max", str(args.vote_y_max)])
    
    if args.vote_x_ticks:
        cmd.extend(["--x_ticks"] + [str(tick) for tick in args.vote_x_ticks])
    
    if args.vote_hierarchical_sampling:
        cmd.append("--hierarchical_sampling")
    
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
        "--initial_color", args.plot_initial_color,
        "--num_total_bins", str(args.num_bins)
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
        
        "--conda_env_path", args.conda_env_path,
        "--slurm_account", args.slurm_account,
        "--slurm_output", args.slurm_output,
        "--slurm_error", args.slurm_error,
        "--slurm_nodes", str(args.slurm_nodes),
        "--slurm_gpus_per_task", str(args.slurm_gpus_per_task),
        "--slurm_tasks", str(args.slurm_tasks),
        "--slurm_cpus_per_task", str(args.slurm_cpus_per_task),
        "--slurm_time", args.slurm_time,
        "--slurm_partition", args.slurm_partition,
        "--random_seed", str(args.random_seed),
        "--check_interval", str(args.check_interval),
        "--max_workers", str(args.max_workers),
        "--prediction_num_model", str(args.prediction_num_model),
        "--prediction_num_seed", str(args.prediction_num_seed)
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
        "--config_path", args.config_file
    ]
    
    if run_command(plot_cmd, "result plotting") != 0:
        logging.error("Stage 05 failed")
        return False
    

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

def print_welcome():
    """Print welcome message and citation information"""
    print("=" * 80)
    print("""
    █████╗ ███████╗       ██████╗██╗      █████╗ ███████╗███████╗ ██████╗ 
   ██╔══██╗██╔════╝      ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██╔═══██╗
   ███████║█████╗  █████╗██║     ██║     ███████║███████╗█████╗  ██║   ██║
   ██╔══██║██╔══╝  ╚════╝██║     ██║     ██╔══██║╚════██║██╔══╝  ██║▄▄ ██║
   ██║  ██║██║           ╚██████╗███████╗██║  ██║███████║███████╗╚██████╔╝
   ╚═╝  ╚═╝╚═╝            ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝ ╚══▀▀═╝ 
    """)
    print("=" * 80)
    print("Welcome to the AF-ClaSEQ pipeline!")
    print("=" * 80)
    print("If you found this tool useful in your publication, please cite the paper:")
    print()
    print("Leveraging Sequence Purification for Accurate Prediction of Multiple")
    print("Conformational States with AlphaFold2")
    print("Enming Xing, Junjie Zhang, Shen Wang, Xiaolin Cheng")
    print()
    print("arXiv:2503.00165 [q-bio.BM]")
    print("https://doi.org/10.48550/arXiv.2503.00165")
    print("=" * 80)
    print()


def main():

    args = parse_args()
    
    print_welcome()
    
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
    if "01_RUN" in stages_to_run:
        if not stage_01_run_iterative_shuffling(args):
            logging.error("Stopping pipeline due to failure in stage 01_RUN")
            return
    
    if "01_ANALYSIS" in stages_to_run:
        if not stage_01_analysis_iterative_shuffling(args):
            logging.error("Stopping pipeline due to failure in stage 01_ANALYSIS")
            return
    
    if "02_RUN" in stages_to_run:
        if not stage_02_run_m_fold_sampling(args):
            logging.error("Stopping pipeline due to failure in stage 02_RUN")
            return
    
    if "02_ANALYSIS" in stages_to_run:
        if not stage_02_analysis_m_fold_sampling(args):
            logging.error("Stopping pipeline due to failure in stage 02_ANALYSIS")
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