# Divide-and-Conquer Workflow

Splits a large MSA into phylogenetically similar groups (clades) and predicts structures for many random sequence combinations within each clade.

[← README](../README.md) · [Workflows](divide_and_conquer.md) · [LOO](leave_one_out.md) · [M-Fold](m_fold_sampling_voting.md) · [Occurrence](occurrence_voting.md) · [UMAP](umap_voting.md) · [Config](configuration.md) · [Troubleshooting](troubleshooting.md)

**When to use**: You have a large MSA (>100 sequences) and want to predict structures systematically.

**What it does**: Splits your MSA into phylogenetically similar groups (clades), then predicts structures for multiple random combinations within each clade. This explores sequence diversity while maintaining evolutionary relationships.

## Before You Start

**You need**:
1. ✅ A multiple sequence alignment file (`.a3m` format)
2. ✅ FastTree binary installed and accessible
3. ✅ ColabFold environment set up
4. ✅ Access to SLURM cluster with GPUs

**Optional**:
- Reference PDB structures if you want to calculate RMSD/TM-score metrics
- Structure analysis config JSON (see example below)

## Step 1: Prepare Your Configuration

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

## Step 2: Test Your Configuration

Before running the full workflow, validate your config:

```bash
python scripts/run_divide_and_conquer.py --config my_protein_config.yaml --dry-run
```

**What to check**:
- ✅ "Configuration validation completed successfully"
- ✅ All file paths are found
- ✅ FastTree binary is accessible

## Step 3: Run the Workflow

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

## Step 4: Check Your Results

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

## If Something Goes Wrong

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

---
[← Back to README](../README.md)
