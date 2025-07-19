# AF-ClaSeq: Leveraging Sequence Purification for Accurate Prediction of Multiple Conformational States with AlphaFold2

AF-ClaSeq is a comprehensive pipeline for protein structure prediction and analysis that leverages sequence purification to accurately predict multiple conformational states using AlphaFold2. The framework includes advanced analysis tools for structure comparison and 3D visualization.

a note here: the results_updates are the new results for NSMB revision, and will be excuted by the newly updated hit_expand pipeline

## Overview

AlphaFold2 has revolutionized protein structure prediction by utilizing co-evolutionary information embedded in multiple sequence alignments (MSAs). AF-ClaSeq extends this capability by systematically isolating co-evolutionary signals through sequence purification and iterative enrichment. The pipeline extracts sequence subsets that preferentially encode distinct structural states, enabling high-confidence predictions of alternative conformations.

Rather than relying solely on MSA depth, AF-ClaSeq focuses on sequence purity to successfully sample alternative states. Our research has revealed that sequences encoding specific structural states are distributed across phylogenetic clades and superfamilies, not limited to specific lineages.

## Key Features

### Core Pipeline
- **Hit Expansion**: Uses MMseqs2 clustering and BLOSUM62-based similarity search to systematically expand and optimize sequence sets
- **M-fold Sampling**: Creates and analyzes multiple sequence groups to explore conformational landscapes
- **Sequence Voting**: Identifies sequences that consistently contribute to specific conformational states
- **Recompilation & Prediction**: Generates purified MSAs for targeted structure prediction
- **Comprehensive Visualization**: Provides detailed analysis tools across all pipeline stages

### Analysis Tools
- **Structure Comparison**: TM-align based comparison between predicted and experimental structures
- **3D Visualization**: Interactive py3Dmol-based structure viewers with side-by-side and overlay modes
- **Statistical Analysis**: Comprehensive plotting with SVG vector graphics for publication-quality figures
- **Performance Metrics**: TM-score, RMSD, pLDDT, and custom structural metrics

## Installation

### Prerequisites

- Python 3.10+
- SLURM-enabled compute cluster (for large-scale predictions)
- ColabFold installation (see [ColabFold repository](https://github.com/sokrypton/ColabFold))
- TM-align program (available from [TM-align website](https://zhanggroup.org/TM-align/))

### Installation Options

#### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/enmingxing244/AF_ClaSeq.git
cd AF_ClaSeq

# Install with Poetry
poetry install
```

#### Using pip

```bash
# Clone the repository
git clone https://github.com/enmingxing244/AF_ClaSeq.git
cd AF_ClaSeq

# Install with pip
pip install -e .
```

#### Additional Dependencies for Analysis Tools

```bash
# Install analysis dependencies
pip install py3Dmol pandas matplotlib seaborn numpy
```

## Pipeline Workflow

AF-ClaSeq consists of six main components:

### Core Pipeline Stages
1. **Hit Expand** (`01_HIT_EXPAND_RUN` & `01_HIT_EXPAND_ANALYSIS`): Uses MMseqs2 clustering and similarity-based expansion to optimize sequence sets
2. **M-fold Sampling** (`02_M_FOLD_SAMPLING_RUN` & `02_M_FOLD_SAMPLING_PLOT`): Performs multiple rounds of sequence sampling
3. **Sequence Voting** (`03_VOTING_RUN`): Analyzes which sequences contribute to specific conformational states
4. **Recompilation & Prediction** (`04_RECOMPILE_PREDICT_RUN`): Recompiles selected sequences and predicts structures
5. **Analysis & Visualization** (`05_PURE_SEQ_PLOT_RUN`): Creates comprehensive visualizations

### Analysis Tools
6. **Structure Comparison**: Post-prediction analysis comparing results with experimental structures

## Usage

### Basic Pipeline Execution

The pipeline is executed using the `run_af_claseq_pipeline.py` script with a YAML configuration file:

```bash
python scripts/run_af_claseq_pipeline.py config_run.yaml
```

### Structure Analysis

For post-prediction analysis, use the structure comparison tool:

```bash
cd analysis_directory
python structure_comparison.py
```

This will automatically:
- Compare predicted structures with experimental references
- Generate TM-align based similarity metrics
- Create statistical plots and 3D visualizations
- Save results in CSV format and interactive HTML files

## Configuration Files

AF-ClaSeq requires two main configuration files:

1. **YAML Configuration File** (`config_run.yaml`): Controls pipeline execution parameters
2. **JSON Filter Configuration** (`config.json`): Defines structural metrics and filters

### YAML Configuration File Structure

The YAML configuration file is organized into the following sections:

```yaml
# Main configuration sections
general:                   # Basic parameters and file paths
slurm:                     # SLURM job submission parameters
pipeline_control:          # Stages to execute and control parameters
hit_expand:                # Parameters for hit expansion stage
m_fold_sampling:           # Parameters for M-fold sampling stage
sequence_voting:           # Parameters for sequence voting stage
recompile_predict:         # Parameters for recompilation and prediction stage
pure_sequence_plotting:    # Parameters for analysis and visualization
```

#### **1. General Section**
Controls basic pipeline parameters and file paths:

```yaml
general:
  # Input files
  source_a3m: "path/to/input.a3m"              # Input MSA file for the pipeline
  default_pdb: "path/to/reference.pdb"         # Reference PDB structure for analysis
  config_file: "path/to/config.json"           # JSON file containing filter criteria definitions
  
  # Output configuration
  base_dir: "run_directory"                    # Base directory for all pipeline outputs
  protein_name: "MyProtein"                    # Name of the protein being analyzed
  
  # Analysis parameters
  coverage_threshold: 0.8                      # Minimum coverage required for sequence alignment (0.0-1.0)
  num_models: 1                                # Number of models to generate per prediction
  random_seed: 42                              # Random seed for reproducibility
  num_bins: 30                                 # Number of bins for histogram analysis
  
  # Visualization colors (applied across all stages)
  plot_initial_color: "#87CEEB"                # Starting color for gradients
  plot_end_color: "#FFFFFF"                    # Ending color for gradients
```

**Key Parameters Explained:**
- `source_a3m`: Starting MSA file in A3M format containing homologous sequences
- `default_pdb`: Reference structure used for initial structural metrics calculation
- `config_file`: JSON configuration defining structural metrics and filtering criteria
- `coverage_threshold`: Minimum sequence coverage required (0.5-0.9 recommended)
- `num_models`: ColabFold models per prediction (1-5, balance speed vs accuracy)

#### **2. SLURM Section**
Controls cluster job submission and resource allocation:

```yaml
slurm:
  # Environment
  conda_env_path: "/path/to/conda/env/"         # Path to conda environment with dependencies
  
  # Account and partition
  slurm_account: "your_account"                 # SLURM account for job submission
  slurm_partition: "your_partition"             # SLURM partition to use
  
  # Resource allocation
  slurm_nodes: 1                                # Number of nodes per job
  slurm_gpus_per_task: 1                        # GPUs per task (required for ColabFold)
  slurm_tasks: 1                                # Number of tasks per job
  slurm_cpus_per_task: 4                        # CPU cores per task
  slurm_time: "04:00:00"                        # Maximum job runtime (HH:MM:SS)
  
  # Job output
  slurm_output: "/dev/null"                     # Standard output destination
  slurm_error: "/dev/null"                      # Standard error destination
  
  # Concurrency control
  max_workers: 64                               # Maximum number of concurrent jobs
```

**Resource Guidelines:**
- `slurm_gpus_per_task`: Always set to 1 for ColabFold jobs
- `slurm_cpus_per_task`: 4-8 cores recommended per GPU
- `slurm_time`: 2-4 hours sufficient for most predictions
- `max_workers`: Adjust based on cluster capacity and fair usage

#### **3. Pipeline Control Section**
Defines which stages to execute and monitoring parameters:

```yaml
pipeline_control:
  # Pipeline stages to execute (comment out to skip)
  stages:
    - "01_HIT_EXPAND_RUN"        # Run hit expansion
    - "01_HIT_EXPAND_ANALYSIS"   # Analyze hit expansion results
    - "02_M_FOLD_SAMPLING_RUN"   # Run M-fold sampling
    - "02_M_FOLD_SAMPLING_PLOT"  # Plot M-fold sampling results
    - "03_VOTING_RUN"            # Run sequence voting
    - "04_RECOMPILE_PREDICT_RUN" # Recompile and predict structures
    - "05_PURE_SEQ_PLOT_RUN"     # Generate plots
  
  check_interval: 60             # Interval (seconds) to check job status
```

**Stage Dependencies:**
- Each stage depends on completion of previous stages
- Comment out stages to resume from a specific point
- Analysis stages can be re-run independently for different parameters

#### **4. Hit Expand Section**
Controls the core sequence expansion and optimization process:

```yaml
hit_expand:
  # Input configuration
  input_msa: ""                               # Leave empty to use general.source_a3m
  
  # Multi-round expansion
  rounds: 1                                   # Number of iterative expansion rounds
  cumulative_expansion: true                  # Accumulate sequences across rounds
  
  # MMseqs2 clustering configuration
  mmseqs_bin: "/path/to/mmseqs"              # Path to MMseqs2 binary
  mmseqs_coverage: 0.8                       # Sequence coverage threshold for clustering
  mmseqs_min_seq_id: 0.3                     # Minimum sequence identity for clustering
  mmseqs_cov_mode: 0                         # Coverage mode (0=target, 1=query, 2=shorter)
  mmseqs_cluster_mode: 0                     # Clustering mode (0=greedy, 1=connected components)
  mmseqs_threads: 8                          # Number of threads for MMseqs2
  
  # Subset generation
  num_subsets: 2000                          # Number of random subsets to generate
  num_random_sequences: 8                    # Sequences per subset (excluding query)
  num_batches: 80                            # Number of batches for parallel processing
  
  # BLOSUM62-based similarity search
  similarity_top_k: 100                      # Maximum similar sequences per query
  similarity_threshold: 0.6                  # Minimum BLOSUM62 similarity (0.0-1.0)
  exclude_query_headers: true                # Exclude sequences with query-like headers
  
  # Structure analysis and filtering
  plddt_threshold: 75.0                      # Minimum pLDDT score for filtering
  filter_criteria_threshold: 6               # Threshold value for structural metric
  filter_criteria: "metric_name"             # Which filter criteria to use (from JSON config)
  
  # Job monitoring
  monitor_jobs: true                         # Monitor SLURM jobs
  job_check_interval: 60.0                   # Job status check interval (seconds)
  check_existing_jobs: true                  # Check if jobs already complete
  
  # Processing control (for debugging/partial runs)
  skip_structure_prediction: false           # Skip ColabFold structure prediction
  skip_structure_analysis: false             # Skip structure analysis and filtering
  skip_hit_expansion: false                  # Skip similarity search expansion
  skip_clustering: false                     # Skip MMseqs2 clustering
```

**Key Parameters:**
- `num_subsets`: Higher values = better sampling, longer runtime (1000-5000 typical)
- `num_random_sequences`: Sequences per subset, balance diversity vs computation (6-12)
- `similarity_threshold`: BLOSUM62 similarity cutoff (0.4-0.8 range)
- `filter_criteria`: Must match a criterion name in the JSON configuration

#### **5. M-fold Sampling Section**
Controls multi-round sampling for conformational exploration:

```yaml
m_fold_sampling:
  # Input (auto-detected from hit_expand if available)
  m_fold_samp_input_a3m: "path/to/expanded.a3m"   # Input MSA from hit expansion
  
  # Sampling parameters
  m_fold_group_size: 6                           # Sequences per sampling group
  m_fold_random_select: null                     # Number of random sequences (null = all)
  rounds: 3                                      # Number of sampling rounds
  
  # Quality filtering
  m_fold_plddt_threshold: 0                      # Minimum pLDDT threshold (0 = no filter)
  
  # Visualization
  m_fold_log_scale: true                         # Use log scale for count plots
  m_fold_n_plot_bins: 50                        # Number of bins for histograms
  m_fold_figsize: [10, 5]                       # Figure size (width, height)
  m_fold_show_bin_lines: true                    # Show bin boundaries on plots
```

#### **6. Sequence Voting Section**
Analyzes sequence contributions to specific conformational states:

```yaml
sequence_voting:
  vote_threshold: 0.0                          # Minimum vote threshold
  vote_min_value: 0                            # Minimum metric value for voting
  vote_max_value: 6                            # Maximum metric value for voting
  vote_figsize: [10, 5]                       # Figure size for voting plots
  vote_hierarchical_sampling: false           # Use hierarchical sampling
  use_focused_bins: true                       # Focus on specific metric bins
```

#### **7. Recompile Predict Section**
Controls final sequence recompilation and structure prediction:

```yaml
recompile_predict:
  # Bin selection for recompilation
  bin_numbers_1: [23, 26]                     # Bin numbers for first metric
  bin_numbers_2: [19, 20]                     # Bin numbers for second metric
  combine_bins: false                          # Whether to combine bins
  
  # Metric mapping (must match JSON config names)
  metric_name_1: "metric1_name"               # Name of first metric
  metric_name_2: "metric2_name"               # Name of second metric (optional)
  
  # Prediction parameters
  prediction_num_model: 5                     # Number of models per prediction
  prediction_num_seed: 8                      # Number of seeds per prediction
```

#### **8. Pure Sequence Plotting Section**
Controls final analysis and visualization:

```yaml
pure_sequence_plotting:
  plddt_threshold: 0                          # Quality threshold for plotting
  figsize: [15, 7]                           # Figure size
  dpi: 600                                   # Resolution for plots
  max_workers: 8                             # Parallel workers for analysis
```

### JSON Filter Configuration Structure

The JSON configuration file defines structural metrics and filters for comparing predicted structures:

```json
{
  "basics": {
    "full_index": {"start": 1, "end": 218},
    "local_index": {"start": 50, "end": 100}
  },
  "filter_criteria": [
    {
      "name": "metric_name",
      "type": "metric_type", 
      "method": "filtering_method",
      "additional_parameters": "value"
    }
  ]
}
```

#### **Basics Section**
- `full_index`: Residue range for the entire protein (required)
- `local_index`: Specific region for local calculations (optional)

#### **Filter Criteria Types**

**1. TM-score Comparison**
```json
{
  "name": "tmscore_to_refA",
  "type": "tmscore",
  "method": "above",
  "ref_pdb": "path/to/reference.pdb"
}
```
- **Purpose**: Measures global structural similarity (0-1 scale)
- **method**: "above" (higher TM-score preferred) or "below"
- **Usage**: Comparing to known conformational states

**2. RMSD Calculation**
```json
{
  "name": "rmsd_to_refA", 
  "type": "rmsd",
  "method": "below",
  "superposition_indices": {"start": 1, "end": 100},
  "rmsd_indices": {"start": 1, "end": 100},
  "ref_pdb": "path/to/reference.pdb"
}
```
- **Purpose**: Measures local structural deviation (Ångströms)
- **superposition_indices**: Residues used for structural alignment
- **rmsd_indices**: Residues used for RMSD calculation
- **Usage**: Fine-grained structural comparison

**3. Distance Measurement**
```json
{
  "name": "domain_distance",
  "type": "distance", 
  "method": "above",
  "indices": {
    "set1": [10, 11, 12, 13],
    "set2": [100, 101, 102, 103]
  }
}
```
- **Purpose**: Measures distance between domain centers
- **Usage**: Detecting domain movements, conformational changes

**4. Angle Measurement**
```json
{
  "name": "domain_angle",
  "type": "angle",
  "method": "above", 
  "indices": {
    "domain1": [10, 11, 12, 13],
    "domain2": [100, 101, 102, 103], 
    "hinge": [50, 51, 52]
  }
}
```
- **Purpose**: Measures inter-domain angles
- **Usage**: Characterizing hinge movements, conformational flexibility

**5. All-Atom RMSD**
```json
{
  "name": "all_atom_rmsd",
  "type": "all_atom_rmsd",
  "method": "below",
  "superposition_indices": {"start": 1, "end": 100},
  "rmsd_indices": {"start": 1, "end": 100},
  "ref_pdb": "path/to/reference.pdb"
}
```
- **Purpose**: High-precision structural comparison using all atoms
- **Usage**: Detailed sidechain and backbone analysis

## Example Configurations

### Example 1: Two-State Protein (KaiB)

**YAML Configuration:**
```yaml
general:
  source_a3m: "input/2QKEE_colabfold-8128Seqs.a3m"
  default_pdb: "input/2QKEE_default.pdb"
  base_dir: "kaib_run"
  config_file: "configs/config_2qke_5jyt_tmscore.json"
  protein_name: "KaiB"
  coverage_threshold: 0.8
  num_models: 1

hit_expand:
  num_subsets: 2000
  num_random_sequences: 8
  filter_criteria: "2qke_tmscore"
  plddt_threshold: 75.0
```

**JSON Configuration:**
```json
{
  "basics": {
    "full_index": {"start": 1, "end": 91}
  },
  "filter_criteria": [
    {
      "name": "2qke_tmscore",
      "type": "tmscore", 
      "method": "above",
      "ref_pdb": "ref/2qkeE.pdb"
    },
    {
      "name": "5jyt_tmscore",
      "type": "tmscore",
      "method": "above", 
      "ref_pdb": "ref/5jytA.pdb"
    }
  ]
}
```

### Example 2: Domain Movement Analysis (Adenylate Kinase)

**JSON Configuration:**
```json
{
  "basics": {
    "full_index": {"start": 1, "end": 214}
  },
  "filter_criteria": [
    {
      "name": "1ake_rmsd",
      "type": "rmsd",
      "method": "below",
      "superposition_indices": {"start": 1, "end": 214},
      "rmsd_indices": {"start": 1, "end": 214}, 
      "ref_pdb": "ref/1AKE.pdb"
    },
    {
      "name": "4ake_rmsd", 
      "type": "rmsd",
      "method": "below",
      "superposition_indices": {"start": 1, "end": 214},
      "rmsd_indices": {"start": 1, "end": 214},
      "ref_pdb": "ref/4AKE.pdb"
    }
  ]
}
```

## Analysis Tools

### Structure Comparison Analysis

The pipeline includes a comprehensive structure comparison tool for post-prediction analysis:

```bash
cd CASP16_targets/analysis
python structure_comparison.py
```

**Features:**
- **Automated Structure Matching**: Finds matching predicted/experimental structure pairs
- **TM-align Integration**: Calculates TM-scores and RMSD values
- **Statistical Analysis**: Generates distribution plots and performance metrics
- **3D Visualization**: Creates interactive HTML viewers with py3Dmol
- **SVG Output**: Publication-quality vector graphics

**Output Files:**
- `structure_comparison_results.csv`: Summary statistics table
- `detailed_structure_comparison_results.csv`: Complete results with TM-align output
- `structure_comparison_plots.svg`: Statistical analysis plots
- `detailed_quality_analysis.svg`: Quality assessment plots  
- `*_structure_comparison.html`: Interactive 3D side-by-side viewers
- `*_overlay.html`: Interactive 3D overlay viewers

**Requirements for Analysis Tools:**
```bash
pip install py3Dmol pandas matplotlib seaborn numpy
```

### Understanding Results

After running the pipeline, results are organized in the specified base directory:

```
run/
├── 01_hit_expand/               # Hit expansion results
│   ├── 01_clustering/           # MMseqs2 clustering output
│   ├── 02_subsets/              # Random subset generation
│   ├── 03_similarity_search/    # BLOSUM62 expansion results
│   ├── 04_prediction/           # ColabFold structure predictions
│   ├── 05_analysis/             # Structure analysis and filtering
│   └── plots/                   # Hit expansion visualizations
├── 02_m_fold_sampling/          # M-fold sampling results
│   ├── round_*/                 # Results for each sampling round
│   └── plot/                    # M-fold sampling plots
├── 03_voting/                   # Sequence voting results
│   └── [metric_name]/           # Results for each metric
├── 04_recompile/                # Recompiled sequences and predictions
│   └── [metric_name]/           # Predictions for selected bins
├── 05_plots/                    # Final analysis plots
└── logs/                        # Pipeline execution logs
```

**Key Output Files:**
- **Hit Expansion Plots**: `01_hit_expand/plots/` - TM-score distributions and filtering results
- **M-fold Sampling**: `02_m_fold_sampling/plot/` - Conformational space exploration
- **Sequence Voting**: `03_voting/[metric]/` - Sequence contribution analysis
- **Final Structures**: `04_recompile/[metric]/prediction/` - Purified sequence predictions
- **Comparative Analysis**: `05_plots/[metric]/` - Final comparative visualizations

### Interpreting Structural Metrics

**TM-score**
- Range: 0.0 to 1.0
- > 0.5: Same fold
- > 0.7: High similarity
- > 0.9: Near-identical structures

**RMSD**
- Units: Ångströms (Å)
- < 2.0 Å: High similarity
- 2.0-4.0 Å: Moderate similarity
- > 4.0 Å: Low similarity

**pLDDT**
- Range: 0 to 100
- > 90: Very high confidence
- 70-90: High confidence  
- 50-70: Low confidence
- < 50: Very low confidence

## Best Practices

### Selecting Metrics

**For conformational state analysis:**
- Use TM-score with known reference structures
- Combine with pLDDT filtering (threshold: 70-80)

**For domain movement studies:**
- Use RMSD with domain-specific superposition
- Add distance/angle measurements for quantification

**For novel conformation discovery:**
- Start with broad metrics (TM-score to multiple references)
- Refine with specific structural features

### Parameter Optimization

**Hit Expansion:**
- Start with 1000-2000 subsets for testing
- Increase to 5000+ for production runs
- Adjust `similarity_threshold` based on sequence diversity

**M-fold Sampling:**
- Use 3-5 rounds for comprehensive sampling
- Adjust `group_size` based on MSA depth

**Sequence Voting:**
- Focus on bins with clear separation
- Consider multiple metrics for complex systems

## Troubleshooting

### Common Issues

**1. Installation Problems**
```bash
# Check dependencies
python -c "import af_claseq; print('AF-ClaSeq installed successfully')"

# Verify TM-align
TMalign -h

# Test ColabFold
colabfold_batch --help
```

**2. Configuration Errors**
- Verify all file paths are absolute and accessible
- Check JSON syntax using online validators
- Ensure metric names match between YAML and JSON configs

**3. SLURM Job Failures**
- Check account and partition settings
- Verify conda environment path
- Monitor job logs in `logs/` directory

**4. Memory Issues**
- Reduce `max_workers` for large jobs
- Increase `slurm_time` for complex proteins
- Use smaller `num_subsets` for initial testing

**5. Structure Analysis Issues**
- Verify reference PDB file formats
- Check residue numbering consistency
- Ensure structural alignment regions are valid

### Performance Optimization

**For large proteins (>500 residues):**
- Increase SLURM time allocation
- Use fewer concurrent workers
- Consider domain-based analysis

**For high-throughput analysis:**
- Optimize `num_batches` for cluster capacity
- Use SSD storage for temporary files
- Implement checkpointing for long runs

## Citation

If you use AF-ClaSeq in your research, please cite:

> Xing, E., Zhang, J., Wang, S., Cheng, X. (2025). Leveraging Sequence Purification for Accurate Prediction of Multiple Conformational States with AlphaFold2. *arXiv preprint* arXiv:2503.00165.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or suggestions:
- **Email**: xing.244@osu.edu
- **GitHub Issues**: [AF-ClaSeq Issues](https://github.com/enmingxing244/AF_ClaSeq/issues)

## Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process