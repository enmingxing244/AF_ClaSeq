# M-Fold Sampling & Voting Workflow

Performs thousands of random sequence-combination predictions to map the conformational landscape, votes to select optimal bins, then re-predicts those bins in detail.

[← README](../README.md) · [Workflows](divide_and_conquer.md) · [LOO](leave_one_out.md) · [M-Fold](m_fold_sampling_voting.md) · [Occurrence](occurrence_voting.md) · [UMAP](umap_voting.md) · [Config](configuration.md) · [Troubleshooting](troubleshooting.md)

**When to use**: You want to explore the conformational landscape and find which sequence combinations produce specific structural states.

**What it does**: Performs thousands of random predictions with different sequence combinations, analyzes which combinations yield desired conformations, then votes to select optimal bins for detailed prediction.

## Before You Start

**You need**:
1. ✅ Source MSA file (`.a3m` format)
2. ✅ ColabFold environment
3. ✅ Reference PDB structures
4. ✅ Structure analysis config JSON


**Important**: This workflow runs in **stages**. You control which stages run via the config file, not command-line arguments.

## Step 1: Prepare Your Configuration

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

## Step 2: Configure Sampling Parameters

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

## Step 3: Choose Which Stages to Run

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

## Step 4: Run Stage 1 - M-Fold Sampling

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

## Step 5: Run Stage 2 - Analyze and Plot

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

## Step 6: Run Stage 3 - Sequence Voting

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

## Step 7: Run Stage 4 - Predict Selected Bins

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

## Step 8: Run Stage 5 - Final Plots

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

## Understanding the Results

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

## Common Issues

**Stage 1 taking forever**:
- Reduce `rounds` from 3 to 2
- Increase `m_fold_group_size` from 10 to 15 (fewer total groups)

**No good bins in voting**:
- Adjust `vote_min_value` and `vote_max_value` ranges
- Try different `vote_threshold` (e.g., 0.25 instead of 0.18)

**Ran wrong stage**:
- Just comment it out in config and run again - completed stages are skipped

---
[← Back to README](../README.md)
