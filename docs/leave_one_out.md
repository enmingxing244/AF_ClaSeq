# Leave-One-Out Validation Workflow

Identifies which sequences are critical to a structure prediction by systematically removing one sequence at a time from each group and measuring the impact on prediction quality.

[← README](../README.md) · [Workflows](divide_and_conquer.md) · [LOO](leave_one_out.md) · [M-Fold](m_fold_sampling_voting.md) · [Occurrence](occurrence_voting.md) · [UMAP](umap_voting.md) · [Config](configuration.md) · [Troubleshooting](troubleshooting.md)

**When to use**: You want to find out which sequences are critical for getting good structure predictions.

**What it does**: Creates groups of sequences, then systematically removes one sequence at a time to see how it affects the prediction quality. Sequences whose removal significantly changes results are identified as "impactful."

## Before You Start

**You need**:
1. ✅ A source MSA file (`.a3m` format) - usually from divide-and-conquer results or a curated set
2. ✅ ColabFold environment set up
3. ✅ Reference PDB structures (to calculate impact metrics)
4. ✅ Structure analysis config JSON file (defines which metrics to calculate)

**Important**: This workflow requires **`num_models: 5`** or higher for reliable results!

## Step 1: Prepare Structure Analysis Config

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

## Step 2: Prepare Your Configuration

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

## Step 3: Test Your Configuration

```bash
python scripts/run_leave_one_out.py --validate-only my_loo_config.yaml
```

**What to check**:
- ✅ Source A3M file exists
- ✅ Structure analysis config JSON is valid
- ✅ All reference PDB files are found

## Step 4: Run the Workflow

```bash
python scripts/run_leave_one_out.py my_loo_config.yaml
```

**What happens**:
1. **Group Creation**: Creates random groups from your sequences
   - Output: `base_dir/groups/group_001/`, `group_002/`, etc.

2. **Leave-One-Out Setup**: For each group, creates subsets
   - Each subset is missing exactly one sequence
   - Output: `group_001_full.a3m`, `group_001_loo_1.a3m`, `group_001_loo_2.a3m`, etc.

3. **ColabFold Prediction**: Submits SLURM jobs for all groups and LOO subsets
   - Output: PDB files in group directories
   - Monitor: `squeue -u $USER`

4. **Impact Analysis**: Calculates impact score for each sequence
   - Compares full group metric vs. LOO subset metric
   - Output: `base_dir/results/impact_analysis.csv`

5. **Plotting**: Generates impact visualizations
   - Output: `base_dir/plots/`

## Step 5: Analyze Results

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

## If ColabFold Jobs Are Already Done

If predictions are complete but you want to re-run only the analysis:

```bash
python scripts/run_leave_one_out.py --analysis-only my_loo_config.yaml
```

This skips group creation and ColabFold, only runs impact analysis.

## Common Issues

**"No significant sequences found"**:
- Your `impact_threshold` might be too high - try lowering from 1.0 to 0.5
- Check if predictions completed successfully

**Too many jobs**:
- The workflow creates N+1 jobs per group (1 full + N LOO subsets)
- If you have 50 groups of 8 sequences = 450 jobs total
- Reduce number of source sequences or increase `num_seq_per_group`

## Next Step: Use Results in Occurrence Voting

The significant sequences from LOO can be used as input for [Occurrence Voting](occurrence_voting.md):

```bash
# Use these sequences for occurrence voting
cp base_dir/results/significant_impact_sequences.a3m input_for_occurrence_voting.a3m
```

---
[← Back to README](../README.md)
