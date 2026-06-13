# Occurrence Voting Workflow

Creates thousands of random sequence groups, predicts each, filters by quality, then selects the sequences that appear most frequently in high-quality structures.

[← README](../README.md) · [Workflows](divide_and_conquer.md) · [LOO](leave_one_out.md) · [M-Fold](m_fold_sampling_voting.md) · [Occurrence](occurrence_voting.md) · [UMAP](umap_voting.md) · [Config](configuration.md) · [Troubleshooting](troubleshooting.md)

**When to use**: You have a pool of sequences (e.g., from Leave-One-Out) and want to find which ones most frequently appear in high-quality predictions.

**What it does**: Creates thousands of random groups, predicts structures for each, filters by quality, then counts how often each sequence appears in good structures. The most frequent sequences are selected as optimal.

## Before You Start

**You need**:
1. ✅ Source MSA file - typically from Leave-One-Out workflow
2. ✅ ColabFold environment
3. ✅ Reference PDB structures
4. ✅ Structure analysis config JSON
5. ✅ Clear quality criteria (which conformational state you want)

## Step 1: Prepare Your Configuration

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

## Step 2: Configure Sampling

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

## Step 3: Set Quality Filter

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

## Step 4: Configure Voting

```yaml
voting:
  top_n_sequences: 16      # How many sequences to select
```

This will select the 16 sequences that appear most frequently in structures passing your quality filter.

## Step 5: Test Configuration

```bash
python scripts/run_occurrence_voting.py --validate-only my_voting_config.yaml
```

**What to check**:
- ✅ Source A3M exists
- ✅ Structure analysis JSON is valid
- ✅ All settings are reasonable

## Step 6: Run Complete Workflow

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

## Step 7: Analyze Results

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

## Alternative: Run Step-by-Step

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

## Common Issues

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

## Next Steps: Use Selected Sequences

The final sequences can be used for:

```bash
# Use for M-Fold Sampling to explore conformational diversity
cp base_dir/occurrence_voting/final_top_16_sequences.a3m input_for_mfold.a3m

# Use for final high-quality predictions with many models
# Create a config for these sequences with num_models=5, num_seeds=10
```

See the [M-Fold Sampling & Voting](m_fold_sampling_voting.md) workflow to continue.

---
[← Back to README](../README.md)
