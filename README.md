# AF_ClaSeq: Leveraging Sequence Purification for Accurate Prediction of Multiple Conformational States with AlphaFold2

A comprehensive bioinformatics toolkit for phylogenetically-guided protein structure prediction using AlphaFold and ColabFold. AF_ClaSeq implements advanced sequence analysis strategies including divide-and-conquer phylogenetic splitting, leave-one-out validation, m-fold sampling, and occurrence voting to improve structure prediction quality.

## 🚀 Features

### Core Analysis Modules
- **Divide-and-Conquer**: Phylogenetically-guided sequence clustering and structure prediction
- **Leave-One-Out**: Cross-validation framework for structure prediction assessment
- **M-Fold Sampling**: Statistical sampling approach for robust structure generation
- **Occurrence Voting**: Consensus-based sequence selection and prediction

### Key Capabilities
- **Intelligent Sequence Clustering**: Distance-guided phylogenetic clade detection
- **Concurrent Job Management**: SLURM-based high-throughput processing
- **Coverage Filtering**: Quality-based sequence filtering with configurable thresholds
- **Smart Grouping**: Optimized sequence group generation avoiding small remainders
- **Comprehensive Analysis**: Statistical analysis and visualization of prediction results


## 🛠️ Installation

### Prerequisites

Before installing AF_ClaSeq, ensure you have the following external tools properly installed:

1. **FastTree** - Required for phylogenetic tree construction
   - Download from: https://morgannprice.github.io/fasttree/
   - For detailed installation instructions, refer to the FastTree documentation

2. **TM-align** - Required for structure alignment and comparison
   - Download from: https://zhanggroup.org/TM-align/
   - For detailed installation instructions, refer to the TM-align documentation

3. **ColabFold** - Required for AlphaFold structure prediction
   - Installation guide: https://github.com/sokrypton/ColabFold
   - **IMPORTANT**: Ensure ColabFold is properly installed and configured with GPU access before proceeding
   - Verify your installation by running ColabFold test predictions

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/enmingxing244/AF_ClaSeq.git
cd AF_ClaSeq
```

2. **Install the AF_ClaSeq package:**

The package dependencies are defined in `pyproject.toml`. Install the package in editable mode:

```bash
pip install -e .
```

This will automatically install all required Python dependencies including:
- ete3 (phylogenetic tree handling)
- biopython (sequence processing)
- numpy, pandas (data analysis)
- matplotlib, seaborn (visualization)
- pyyaml (configuration management)

3. **Verify installation:**
```bash
# Check that AF_ClaSeq modules are importable
python -c "import af_claseq; print('AF_ClaSeq successfully installed!')"

# Verify external tools are accessible
which FastTree
which TMalign  # or TMscore, depending on your TM-align installation
```

4. **Configure paths in your config files:**

Update the configuration files to point to your ColabFold environment and FastTree binary:
```yaml
input:
  a3m_file: "/path/to/input/alignment.a3m"
  fasttree_binary: "/path/to/FastTree"

output:
  working_dir: "/path/to/output/working_directory"

colabfold:
  conda_env: "/path/to/colabfold/conda/env"
```

## 📖 Usage

AF_ClaSeq provides four main workflows for protein structure prediction and sequence analysis. This guide will walk you through each workflow step-by-step, from preparation to execution.

> **💡 Tip**: Start with the example configurations in `example/config_examples/` and modify them for your protein.

---

## Workflow 1: Divide-and-Conquer

**When to use**: You have a large MSA (>100 sequences) and want to predict structures systematically.

**What it does**: Splits your MSA into phylogenetically similar groups (clades), then predicts structures for multiple random combinations within each clade. This explores sequence diversity while maintaining evolutionary relationships.

### Before You Start

**You need**:
1. ✅ A multiple sequence alignment file (`.a3m` format)
2. ✅ FastTree binary installed and accessible
3. ✅ ColabFold environment set up
4. ✅ Access to SLURM cluster with GPUs

**Optional**:
- Reference PDB structures if you want to calculate RMSD/TM-score metrics
- Structure analysis config JSON (see example below)

### Step 1: Prepare Your Configuration

Copy the example config and edit it:
```bash
cp example/config_examples/divide_and_conquer_config.yaml my_protein_config.yaml
```

Edit these **required** settings in `my_protein_config.yaml`:

```yaml
input:
  a3m_file: "/path/to/your_protein.a3m"              # Your alignment file
  fasttree_binary: "/path/to/FastTree"                # Where FastTree is installed

output:
  working_dir: "/path/to/output/my_protein_dac"      # Where results will be saved

colabfold:
  conda_env: "/path/to/your/colabfold/env"           # Your ColabFold environment

slurm:
  account: "your_slurm_account"                       # Your SLURM account
  partition: "your_gpu_partition"                     # GPU partition name
```

**Adjust these settings** based on your needs:

```yaml
clade_splitting:
  min_clade_size: 40       # Smaller = more clades (more jobs)
  max_clade_size: 150      # Larger = fewer clades (fewer jobs)
  coverage_filter:
    enabled: true
    threshold: 0.7         # Keep only sequences with ≥70% non-gap characters

shuffling:
  num_shuffles: 10         # More = better sampling but more compute
  group_size: 8            # Typical: 8-15 sequences per group
```

### Step 2: Test Your Configuration

Before running the full workflow, validate your config:

```bash
python scripts/run_divide_and_conquer.py --config my_protein_config.yaml --dry-run
```

**What to check**:
- ✅ "Configuration validation completed successfully"
- ✅ All file paths are found
- ✅ FastTree binary is accessible

### Step 3: Run the Workflow

```bash
python scripts/run_divide_and_conquer.py --config my_protein_config.yaml
```

**What happens** (5 stages):
1. **Phylogenetic Processing** (~5-30 min): Builds tree, splits into clades
   - Output: `working_dir/tree.nwk`, `working_dir/clades/`

2. **Shuffle Management** (~1-5 min): Creates random sequence groups
   - Output: `working_dir/clades/clade_*/shuffle_*/`

3. **ColabFold Prediction** (hours to days): Submits SLURM jobs
   - Output: PDB files in shuffle directories
   - Monitor: `squeue -u $USER` to see running jobs

4. **Structure Analysis** (~10-30 min): Calculates metrics
   - Output: `working_dir/structure_analysis_results.csv`

5. **Plot Generation** (~5-10 min): Creates visualizations
   - Output: `working_dir/plots/`

### Step 4: Check Your Results

After completion, check:

```bash
# Check log file
tail -100 working_dir/logs/workflow.log

# Check analysis results
head working_dir/structure_analysis_results.csv

# View plots
ls working_dir/plots/
```

**Expected outputs**:
- `working_dir/clades/` - Individual clade alignments
- `working_dir/clades/clade_*/shuffle_*/` - PDB structure files
- `working_dir/structure_analysis_results.csv` - Metrics for all structures
- `working_dir/plots/` - Visualization PDFs

### If Something Goes Wrong

**Job failed during ColabFold stage**:
```bash
# Resume from ColabFold predictions (skips steps 1-2)
python scripts/run_divide_and_conquer.py --config my_protein_config.yaml --resume-from step3
```

**Not enough sequences after filtering**:
- Lower `coverage_filter.threshold` from 0.7 to 0.5 in your config

**Too many jobs submitted**:
- Reduce `colabfold.max_concurrent_jobs` from 80 to 40
- Increase `clade_splitting.max_clade_size` to create fewer clades

## Workflow 2: Leave-One-Out Validation

**When to use**: You want to find out which sequences are critical for getting good structure predictions.

**What it does**: Creates groups of sequences, then systematically removes one sequence at a time to see how it affects the prediction quality. Sequences whose removal significantly changes results are identified as "impactful."

### Before You Start

**You need**:
1. ✅ A source MSA file (`.a3m` format) - usually from divide-and-conquer results or a curated set
2. ✅ ColabFold environment set up
3. ✅ Reference PDB structures (to calculate impact metrics)
4. ✅ Structure analysis config JSON file (defines which metrics to calculate)

**Important**: This workflow requires **`num_models: 5`** or higher for reliable results!

### Step 1: Prepare Structure Analysis Config

Create a JSON file that defines your reference structures and metrics:

```bash
# Create my_protein_analysis.json
cat > my_protein_analysis.json << 'EOF'
{
  "reference_structures": [
    {
      "name": "active_state",
      "pdb_file": "/path/to/active_conformation.pdb",
      "chain": "A"
    },
    {
      "name": "inactive_state",
      "pdb_file": "/path/to/inactive_conformation.pdb",
      "chain": "A"
    }
  ],
  "filter_criteria": [
    {
      "name": "active_state_rmsd",
      "metric_type": "composite_rmsd",
      "reference": "active_state"
    },
    {
      "name": "inactive_state_rmsd",
      "metric_type": "composite_rmsd",
      "reference": "inactive_state"
    }
  ]
}
EOF
```

### Step 2: Prepare Your Configuration

Copy and edit the example config:

```bash
cp example/config_examples/leave_one_out_config.yaml my_loo_config.yaml
```

Edit these **required** settings:

```yaml
general:
  source_a3m: "/path/to/your_sequences.a3m"            # Input MSA
  base_dir: "/path/to/output/my_protein_loo"          # Output directory
  structure_analysis_config: "/path/to/my_protein_analysis.json"  # Your JSON file
  protein_name: "MyProtein"

leave_one_out:
  num_seq_per_group: 8                                 # Group size
  impact_metric_name: "active_state_rmsd"              # Which metric to use (must match JSON)
  impact_threshold: 1.0                                # How much change = "significant"
  cutoff_method: "above"                               # "above" or "below"
```

**Understanding cutoff_method**:
- Use `"above"` if **removing** the sequence makes the metric **worse** (higher = more impact)
  - Example: RMSD increases by >1.0 → sequence was helpful
- Use `"below"` if you want the opposite logic

```yaml
slurm:
  conda_env_path: "/path/to/colabfold/env"
  account: "your_slurm_account"
  partition: "your_gpu_partition"
  num_models: 5              # ⚠️ MUST be ≥5 for reliable mean differences!
  max_concurrent_jobs: 200   # Can be high - these are small jobs
```

### Step 3: Test Your Configuration

```bash
python scripts/run_leave_one_out.py --validate-only my_loo_config.yaml
```

**What to check**:
- ✅ Source A3M file exists
- ✅ Structure analysis config JSON is valid
- ✅ All reference PDB files are found

### Step 4: Run the Workflow

```bash
python scripts/run_leave_one_out.py my_loo_config.yaml
```

**What happens**:
1. **Group Creation** (~1-5 min): Creates random groups from your sequences
   - Output: `base_dir/groups/group_001/`, `group_002/`, etc.

2. **Leave-One-Out Setup** (~5-15 min): For each group, creates subsets
   - Each subset is missing exactly one sequence
   - Output: `group_001_full.a3m`, `group_001_loo_1.a3m`, `group_001_loo_2.a3m`, etc.

3. **ColabFold Prediction** (hours): Submits SLURM jobs for all groups and LOO subsets
   - Output: PDB files in group directories
   - Monitor: `squeue -u $USER`

4. **Impact Analysis** (~10-30 min): Calculates impact score for each sequence
   - Compares full group metric vs. LOO subset metric
   - Output: `base_dir/results/impact_analysis.csv`

5. **Plotting** (~5 min): Generates impact visualizations
   - Output: `base_dir/plots/`

### Step 5: Analyze Results

Check which sequences have significant impact:

```bash
# View impact analysis results
head -20 base_dir/results/impact_analysis.csv

# Significant sequences are saved here
cat base_dir/results/significant_impact_sequences.a3m

# Read the summary
cat base_dir/results/sequence_impact_summary.txt
```

**Interpreting results**:
- **Positive impact score**: Removing this sequence made predictions worse → sequence is helpful
- **Negative impact score**: Removing this sequence made predictions better → sequence might be noisy
- **Near-zero impact**: Sequence doesn't significantly affect predictions

### If ColabFold Jobs Are Already Done

If predictions are complete but you want to re-run only the analysis:

```bash
python scripts/run_leave_one_out.py --analysis-only my_loo_config.yaml
```

This skips group creation and ColabFold, only runs impact analysis.

### Common Issues

**"No significant sequences found"**:
- Your `impact_threshold` might be too high - try lowering from 1.0 to 0.5
- Check if predictions completed successfully

**Too many jobs**:
- The workflow creates N+1 jobs per group (1 full + N LOO subsets)
- If you have 50 groups of 8 sequences = 450 jobs total
- Reduce number of source sequences or increase `num_seq_per_group`

### Next Step: Use Results in Occurrence Voting

The significant sequences from LOO can be used as input for Occurrence Voting:

```bash
# Use these sequences for occurrence voting
cp base_dir/results/significant_impact_sequences.a3m input_for_occurrence_voting.a3m
```

## Workflow 3: M-Fold Sampling & Voting

**When to use**: You want to explore the conformational landscape and find which sequence combinations produce specific structural states.

**What it does**: Performs thousands of random predictions with different sequence combinations, analyzes which combinations yield desired conformations, then votes to select optimal bins for detailed prediction.

### Before You Start

**You need**:
1. ✅ Source MSA file (`.a3m` format)
2. ✅ ColabFold environment
3. ✅ Reference PDB structures
4. ✅ Structure analysis config JSON
5. ✅ Patience - this workflow can take days!

**Important**: This workflow runs in **stages**. You control which stages run via the config file, not command-line arguments.

### Step 1: Prepare Your Configuration

Copy and edit the example:

```bash
cp example/config_examples/m_fold_config_run2.yaml my_mfold_config.yaml
```

Edit **required** settings:

```yaml
general:
  source_a3m: "/path/to/your_sequences.a3m"
  base_dir: "/path/to/output/my_protein_mfold"
  config_file: "/path/to/my_protein_analysis.json"  # Structure analysis JSON
  protein_name: "MyProtein"

  # Choose 1 or 2 metrics to analyze
  metric1_name: "active_state_rmsd"    # Must match your JSON file
  metric2_name: "inactive_state_rmsd"  # Optional: for 2D analysis
  num_bins: 30                         # Histogram bins

slurm:
  conda_env_path: "/path/to/colabfold/env"
  slurm_account: "your_account"
  slurm_partition: "your_partition"
  max_workers: 200                     # High! This workflow creates many jobs
```

### Step 2: Configure Sampling Parameters

```yaml
m_fold_sampling:
  m_fold_samp_input_a3m: "/path/to/your_sequences.a3m"  # Usually same as source_a3m
  m_fold_group_size: 10     # Sequences per group (smaller = faster, less context)
  rounds: 3                 # Number of independent sampling rounds
                            # More rounds = better coverage of conformational space
```

**How many predictions will this create?**
- If you have 100 sequences, group_size=10, rounds=3:
- Each round creates ~10 groups (100 sequences / 10 per group)
- Total: 3 rounds × 10 groups = **30 predictions**

### Step 3: Choose Which Stages to Run

The workflow has 4 stages. Edit your config to run them:

```yaml
pipeline_control:
  stages:
    - "01_M_FOLD_SAMPLING_RUN"    # Do random sampling and predictions
    # - "01_M_FOLD_SAMPLING_PLOT"   # Analyze and plot results
    # - "02_VOTING_RUN"             # Vote for best bins
    # - "03_RECOMPILE_PREDICT_RUN"  # Predict selected bins with more models
    # - "04_PURE_SEQ_PLOT_RUN"      # Final analysis plots
```

**Start with only Stage 1**, then uncomment others after each stage completes.

### Step 4: Run Stage 1 - M-Fold Sampling

```bash
python scripts/run_m_fold_sampling_voting.py my_mfold_config.yaml
```

**What happens** (this can take hours/days):
- Creates multiple rounds of random sequence groups
- Submits ColabFold jobs for all groups
- Waits for all predictions to complete
- Output: `base_dir/01_m_fold_sampling/round_1/`, `round_2/`, `round_3/`

Monitor progress:
```bash
# Check how many jobs are running
squeue -u $USER | grep -c "MyProtein"

# Check log file
tail -f base_dir/logs/af_claseq_pipeline.log
```

### Step 5: Run Stage 2 - Analyze and Plot

Once all predictions are done, uncomment the next stage in your config:

```yaml
pipeline_control:
  stages:
    # - "01_M_FOLD_SAMPLING_RUN"    # Already done
    - "01_M_FOLD_SAMPLING_PLOT"   # Now run this
```

```bash
python scripts/run_m_fold_sampling_voting.py my_mfold_config.yaml
```

**What happens** (~10-30 min):
- Calculates metrics for all predictions
- Bins predictions by metric values
- Creates histograms showing distribution
- Output: `base_dir/01_m_fold_sampling/plot/`, `base_dir/01_m_fold_sampling/csv/`

**Check results**:
```bash
# View the histogram plots
ls base_dir/01_m_fold_sampling/plot/

# Check the CSV data
head base_dir/01_m_fold_sampling/csv/histogram_data_*.csv
```

### Step 6: Run Stage 3 - Sequence Voting

Configure voting parameters:

```yaml
sequence_voting:
  vote_threshold: 0.18          # Top 18% of bins
  vote_min_value: 0.4           # Minimum metric value to consider
  vote_max_value: 1.0           # Maximum metric value
```

Uncomment stage in config and run:

```yaml
pipeline_control:
  stages:
    # - "01_M_FOLD_SAMPLING_RUN"
    # - "01_M_FOLD_SAMPLING_PLOT"
    - "02_VOTING_RUN"             # Now run this
```

```bash
python scripts/run_m_fold_sampling_voting.py my_mfold_config.yaml
```

**What happens** (~5-15 min):
- Identifies which bins have most predictions
- Counts sequence occurrences in top bins
- Creates voting plots
- Output: `base_dir/02_voting/metric1_name/`, `metric2_name/`

**Select bins for detailed prediction**:
```bash
# Check voting results
head base_dir/02_voting/active_state_rmsd/voting_results.csv

# Look for bins with high counts in your desired metric range
# For example, if bin 26 has many low-RMSD predictions, use it!
```

### Step 7: Run Stage 4 - Predict Selected Bins

Update config with your chosen bins:

```yaml
recompile_predict:
  bin_numbers_1: [26]           # Bins for metric1 (from voting CSV)
  bin_numbers_2: [9]            # Bins for metric2 (optional)
  combine_bins: false           # Set true to merge bins
  prediction_num_model: 5       # More models for diversity
  prediction_num_seed: 8        # More seeds
```

Uncomment and run:

```yaml
pipeline_control:
  stages:
    - "03_RECOMPILE_PREDICT_RUN"  # Now run this
```

```bash
python scripts/run_m_fold_sampling_voting.py my_mfold_config.yaml
```

**What happens** (hours):
- Extracts sequences from selected bins
- Creates new MSAs with those sequences
- Predicts with more models and seeds for diversity
- Output: `base_dir/03_recompile/metric1_name/bin_26/`

### Step 8: Run Stage 5 - Final Plots

```yaml
pipeline_control:
  stages:
    - "04_PURE_SEQ_PLOT_RUN"      # Final stage
```

```bash
python scripts/run_m_fold_sampling_voting.py my_mfold_config.yaml
```

**What happens** (~10-20 min):
- Analyzes final predictions
- Creates comprehensive plots
- Output: `base_dir/04_plots/`

### Understanding the Results

After all stages complete:

```bash
# 1. Check sampling distribution
ls base_dir/01_m_fold_sampling/plot/

# 2. See which sequences voted for each bin
head base_dir/02_voting/*/voting_results.csv

# 3. Review final predictions
ls base_dir/03_recompile/*/bin_*/

# 4. View final analysis plots
ls base_dir/04_plots/
```

### Common Issues

**Stage 1 taking forever**:
- Reduce `rounds` from 3 to 2
- Increase `m_fold_group_size` from 10 to 15 (fewer total groups)

**No good bins in voting**:
- Adjust `vote_min_value` and `vote_max_value` ranges
- Try different `vote_threshold` (e.g., 0.25 instead of 0.18)

**Ran wrong stage**:
- Just comment it out in config and run again - completed stages are skipped

## Workflow 4: Occurrence Voting

**When to use**: You have a pool of sequences (e.g., from Leave-One-Out) and want to find which ones most frequently appear in high-quality predictions.

**What it does**: Creates thousands of random groups, predicts structures for each, filters by quality, then counts how often each sequence appears in good structures. The most frequent sequences are selected as optimal.

### Before You Start

**You need**:
1. ✅ Source MSA file - typically from Leave-One-Out workflow
2. ✅ ColabFold environment
3. ✅ Reference PDB structures
4. ✅ Structure analysis config JSON
5. ✅ Clear quality criteria (which conformational state you want)

### Step 1: Prepare Your Configuration

Copy and edit the example:

```bash
cp example/config_examples/occurrence_voting.yaml my_voting_config.yaml
```

Edit **required** settings:

```yaml
general:
  source_a3m: "/path/to/significant_impact_sequences.a3m"  # From LOO workflow
  base_dir: "/path/to/output/my_protein_voting"
  protein_name: "MyProtein"

structure_analysis:
  config_json: "/path/to/my_protein_analysis.json"  # Your structure analysis JSON

slurm:
  conda_env_path: "/path/to/colabfold/env"
  account: "your_account"
  partition: "your_partition"
```

### Step 2: Configure Sampling

Decide how many random groups to create:

```yaml
sampling:
  num_groups: 3000         # More = better statistics (typical: 1000-5000)
  group_size: 8            # Sequences per group
  num_batches: 90          # Splits groups into batches for SLURM
```

**How many jobs?**
- 3000 groups = 3000 ColabFold jobs
- If `num_batches: 90`, jobs are organized into 90 batches
- Each batch has ~33 jobs (3000/90)

### Step 3: Set Quality Filter

Define what counts as a "good" structure:

```yaml
filtering:
  metric_name: "active_state_rmsd"  # Which metric to use (from your JSON)
  cutoff_value: 4.0                 # Threshold
  cutoff_method: "below"            # Keep structures with metric < 4.0
```

**Understanding cutoff_method**:
- `"below"`: Keep structures where metric < cutoff (good for RMSD - lower is better)
- `"above"`: Keep structures where metric > cutoff (good for TM-score - higher is better)

### Step 4: Configure Voting

```yaml
voting:
  top_n_sequences: 16      # How many sequences to select
```

This will select the 16 sequences that appear most frequently in structures passing your quality filter.

### Step 5: Test Configuration

```bash
python scripts/run_occurrence_voting.py --validate-only my_voting_config.yaml
```

**What to check**:
- ✅ Source A3M exists
- ✅ Structure analysis JSON is valid
- ✅ All settings are reasonable

### Step 6: Run Complete Workflow

```bash
python scripts/run_occurrence_voting.py my_voting_config.yaml
```

**What happens** (this can take hours to days):

1. **Sampling** (~5-30 min): Creates random groups
   - Output: `base_dir/sampling/batches/`, `groups/`

2. **ColabFold** (hours to days): Predicts all groups
   - Output: `base_dir/colabfold_predictions/batch_*/group_*/`
   - Monitor: `squeue -u $USER`

3. **Analysis** (~30-60 min): Calculates metrics and filters
   - Output: `base_dir/structure_analysis/metrics.csv`
   - Output: `base_dir/structure_analysis/filtered_structures.csv`

4. **Voting** (~10-20 min): Counts occurrences and selects top sequences
   - Output: `base_dir/occurrence_voting/sequence_occurrences.csv`
   - Output: `base_dir/occurrence_voting/final_top_N_sequences.a3m`

Monitor progress:
```bash
# Check running jobs
squeue -u $USER | grep MyProtein

# Watch log file
tail -f base_dir/logs/occurrence_voting.log

# Check how many structures passed filter (updates during analysis)
wc -l base_dir/structure_analysis/filtered_structures.csv
```

### Step 7: Analyze Results

After workflow completes:

```bash
# 1. Check occurrence counts
head -20 base_dir/occurrence_voting/sequence_occurrences.csv

# 2. View top selected sequences
cat base_dir/occurrence_voting/final_top_16_sequences.a3m

# 3. Read summary report
cat base_dir/occurrence_voting/summary_report.txt

# 4. View occurrence plots
ls base_dir/occurrence_voting/occurrence_plots/
```

**Understanding the results**:
- **occurrence_count**: How many times this sequence appeared in good structures
- **occurrence_fraction**: What percentage of good structures contain this sequence
- **Higher count = more reliable** for producing your desired conformation

### Alternative: Run Step-by-Step

If you want more control, run individual steps:

```bash
# Step 1: Just create the random groups
python scripts/run_occurrence_voting.py --sampling-only my_voting_config.yaml

# Step 2: Submit ColabFold jobs (manual or use --step-only colabfold)
python scripts/run_occurrence_voting.py --step-only colabfold my_voting_config.yaml

# Step 3: Run analysis after predictions complete
python scripts/run_occurrence_voting.py --step-only analysis my_voting_config.yaml

# Step 4: Run voting
python scripts/run_occurrence_voting.py --step-only voting my_voting_config.yaml
```

### Common Issues

**No sequences pass quality filter**:
- Your `cutoff_value` might be too strict
- Try relaxing: RMSD from 4.0 to 6.0, or TM-score from 0.7 to 0.6
- Check `base_dir/structure_analysis/metrics.csv` to see actual metric distribution

**Very few structures pass filter**:
- If < 100 structures pass, increase `num_groups` from 3000 to 5000
- Or relax your `cutoff_value`

**All sequences have similar occurrence counts**:
- Increase `num_groups` for better statistics
- Your sequences might all be equally good (or bad) for this conformation

**Out of memory during analysis**:
- The analysis loads all predictions into memory
- Reduce `num_groups` or split into smaller batches

### Next Steps: Use Selected Sequences

The final sequences can be used for:

```bash
# Use for M-Fold Sampling to explore conformational diversity
cp base_dir/occurrence_voting/final_top_16_sequences.a3m input_for_mfold.a3m

# Use for final high-quality predictions with many models
# Create a config for these sequences with num_models=5, num_seeds=10
```

---

## Putting It All Together: Complete Pipeline Example

Here's a recommended workflow from start to finish:

### Scenario: You have a large MSA and want to predict multiple conformational states

**Step 1: Divide-and-Conquer** (1-3 days)
```bash
# Goal: Get initial predictions from large MSA
python scripts/run_divide_and_conquer.py --config my_dac_config.yaml
# Output: Many predictions organized by phylogenetic clades
```

**Step 2: Leave-One-Out** (2-5 days)
```bash
# Goal: Find impactful sequences from promising clade
# Use sequences from best clade as input
python scripts/run_leave_one_out.py my_loo_config.yaml
# Output: significant_impact_sequences.a3m
```

**Step 3: Occurrence Voting** (1-2 days)
```bash
# Goal: Select optimal subset from impactful sequences
# Use significant sequences from LOO as input
python scripts/run_occurrence_voting.py my_voting_config.yaml
# Output: final_top_16_sequences.a3m
```

**Step 4: M-Fold Sampling** (3-5 days)
```bash
# Goal: Explore conformational space with optimal sequences
# Use top sequences from voting as input
python scripts/run_m_fold_sampling_voting.py my_mfold_config.yaml
# Output: High-quality predictions binned by conformation
```

---

##  Workflow Selection Guide

| Starting Point | Goal | Recommended Workflow |
|---------------|------|---------------------|
| Large MSA (>1000 seqs) | Get initial predictions | **Divide-and-Conquer** |
| Curated MSA | Find important sequences | **Leave-One-Out** |
| Pool of sequences | Select optimal subset | **Occurrence Voting** |
| Small set of sequences | Explore conformations | **M-Fold Sampling** |

**Quick Decision Tree**:
1. **Do you have >500 sequences?** → Start with Divide-and-Conquer
2. **Do you know which sequences matter?** → No? Use Leave-One-Out
3. **Do you have too many good sequences?** → Use Occurrence Voting to narrow down
4. **Ready for detailed conformational analysis?** → Use M-Fold Sampling

---

## General Tips and Best Practices

### Before Starting Any Workflow

1. **Test with dry-run**: Always validate your config first
2. **Start small**: Use a subset of sequences for initial testing
3. **Monitor jobs**: Use `squeue -u $USER` and check log files
4. **Check outputs**: Look at intermediate results before proceeding

### Resource Management

```yaml
# For small proteins (<200 residues)
slurm:
  time: "00:30:00"
  cpus: 4

# For medium proteins (200-400 residues)
slurm:
  time: "01:00:00"
  cpus: 8

# For large proteins (>400 residues)
slurm:
  time: "02:00:00"
  cpus: 12
```

### When Jobs Fail

1. **Check the log file** first: `tail -100 base_dir/logs/*.log`
2. **Check SLURM output**: `cat slurm-*.out`
3. **Common fixes**:
   - Increase `slurm.time` if jobs timed out
   - Reduce `max_concurrent_jobs` if queue is full
   - Check ColabFold environment is activated

### Saving Compute Time

- Use `--dry-run` and `--validate-only` modes liberally
- Start with small `num_shuffles`, `rounds`, `num_groups` for testing
- Use `--analysis-only` or `--step-only` to rerun analysis without predictions
- Keep successful predictions - use resume capabilities

---

For complete parameter descriptions and advanced options, see example configuration files in `example/config_examples/`.

## 📁 Project Structure

```
AF_ClaSeq/
├── src/af_claseq/
│   ├── divide_and_conquer/     # Phylogenetic clustering module
│   │   ├── phylogenetic_processor.py
│   │   ├── shuffle_manager.py
│   │   ├── nwk_parse.py
│   │   └── utils.py
│   ├── leave_one_out/          # Cross-validation module
│   ├── m_fold_sampling_voting/ # Statistical sampling module
│   ├── occurrence_voting/      # Consensus voting module
│   └── utils/                  # Shared utilities
│       ├── slurm_utils.py
│       ├── sequence_processing.py
│       └── exceptions.py
├── scripts/                    # Main execution scripts
│   ├── run_divide_and_conquer.py
│   ├── run_leave_one_out.py
│   ├── run_m_fold_sampling_voting.py
│   └── run_occurrence_voting.py
├── examples/                   # Example datasets and configs
│   ├── ABL1/
│   ├── KaiB/
│   └── unified_af_claseq_config.yaml
├── config_examples/           # Configuration templates
└── docs/                      # Documentation
```

## ⚙️ Configuration

### Core Configuration Parameters

#### Input/Output Configuration
```yaml
input:
  a3m_file: "/path/to/input/alignment.a3m"  # Input alignment file
  fasttree_binary: "/path/to/FastTree"      # FastTree executable path

output:
  working_dir: "/path/to/output/directory"  # Working directory for outputs
```

#### Clade Splitting
```yaml
clade_splitting:
  min_clade_size: 40          # Minimum sequences per clade
  max_clade_size: 150         # Maximum sequences per clade
  coverage_filter:
    enabled: true             # Enable coverage-based filtering
    threshold: 0.7            # Minimum non-gap content ratio (70%)
```

#### Sequence Shuffling
```yaml
shuffling:
  num_shuffles: 10            # Number of random shuffles per clade
  group_size: 8               # Number of sequences per group
  random_seed: 42             # For reproducible shuffling
```

#### ColabFold Integration
```yaml
colabfold:
  conda_env: "/path/to/colabfold/conda/env"  # ColabFold environment path
  num_models: 1                              # Number of models to generate
  num_seeds: 1                               # Number of random seeds
  max_concurrent_jobs: 80                    # Maximum parallel jobs
```

#### SLURM Configuration
```yaml
slurm:
  account: "your_slurm_account"   # SLURM account name
  partition: "your_partition"     # SLURM partition name
  time: "01:00:00"                # Job time limit
  memory: "32G"                   # Memory per job
  cpus: 8                         # CPU cores per job
```

## 🔬 Scientific Background

### Divide-and-Conquer Approach
The divide-and-conquer strategy addresses the challenge of processing large multiple sequence alignments (MSAs) by:

1. **Phylogenetic Clustering**: Using FastTree to construct phylogenetic relationships
2. **Distance-Guided Splitting**: Intelligently dividing sequences into coherent clades
3. **Parallel Processing**: Distributing ColabFold predictions across SLURM cluster
4. **Quality Control**: Coverage filtering to remove low-quality sequences

### Coverage Filtering
Sequences are filtered based on gap content to ensure high-quality predictions:
- **Coverage Ratio**: `(uppercase_letters) / (uppercase_letters + gaps)`
- **Configurable Threshold**: Typically 0.7 (70% non-gap content)
- **Quality Improvement**: Removes fragmentary or poorly aligned sequences

### Smart Grouping Algorithm
Optimized sequence grouping prevents small final groups:
- **Remainder Management**: Merges small remainders with previous groups
- **Balanced Distribution**: Maintains reasonably sized groups for prediction
- **Efficiency**: Reduces total number of ColabFold jobs required

## 📊 Output Analysis

### Result Structure
```
results/
├── phylogenetic_analysis/
│   ├── preprocessed.a3m        # Cleaned and filtered alignment
│   ├── tree.nwk               # Phylogenetic tree
│   └── clades/                # Individual clade alignments
├── structure_predictions/
│   ├── clade_001/
│   │   ├── shuffle_01/
│   │   │   ├── group_001.a3m
│   │   │   └── predicted_structures/
│   │   └── shuffle_XX/
│   └── clade_XXX/
├── analysis/
│   ├── statistics.json        # Prediction statistics
│   ├── quality_metrics.csv    # Quality assessment
│   └── plots/                 # Visualization outputs
└── logs/                      # Processing logs
```

### Quality Metrics
- **Prediction Confidence**: Per-residue confidence scores
- **Structural Diversity**: RMSD between predictions
- **Coverage Analysis**: Sequence representation statistics
- **Phylogenetic Coherence**: Clade composition assessment

## 🚨 Troubleshooting

### Common Issues

#### Header Mismatch Errors
```
Error: sequences not found in alignment
```
**Solution**: Header format inconsistency between tree and alignment files has been resolved in v1.2.2.

#### Job Submission Failures
```
Error: SLURM job submission failed
```
**Solutions**:
- Verify SLURM account and partition settings
- Check GPU availability and resource limits
- Ensure ColabFold environment is properly configured

#### Memory Issues
```
Error: FastTree out of memory
```
**Solutions**:
- Increase SLURM memory allocation
- Apply more stringent coverage filtering
- Reduce maximum clade size

#### Coverage Filter Issues
```
Warning: No sequences pass coverage threshold
```
**Solutions**:
- Lower coverage threshold (e.g., from 0.7 to 0.5)
- Check input alignment quality
- Verify sequence preprocessing steps

### Performance Optimization

#### For Large Datasets
- Increase `max_concurrent_jobs` based on cluster capacity
- Use larger `max_clade_size` to reduce total number of jobs
- Enable coverage filtering to reduce sequence count

#### For Small Datasets
- Reduce `min_clade_size` to ensure adequate clustering
- Decrease `num_shuffles` for faster processing
- Consider single-node processing for very small alignments

## 📄 Citation

If you use AF_ClaSeq in your research, please cite:

```
[Citation information to be added upon publication]
```

## 🤝 Contributing

We welcome contributions to AF_ClaSeq! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check existing documentation and examples
- Review troubleshooting section above

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔄 Version History

### v1.2.2 - Latest
- **Major Bug Fixes**: Resolved header mismatch issues in phylogenetic processing
- **Enhanced Job Management**: Direct SlurmJobSubmitter integration
- **Coverage Filtering**: Configurable sequence quality filtering
- **Smart Grouping**: Improved remainder handling in sequence shuffling
- **Performance Improvements**: Optimized concurrent job processing

### Previous Versions
- v1.2.1: M-fold sampling enhancements
- v1.2.0: Leave-one-out validation framework
- v1.1.0: Occurrence voting implementation
- v1.0.0: Initial divide-and-conquer implementation

---

**AF_ClaSeq** - Advancing protein structure prediction through phylogenetically-guided sequence analysis
