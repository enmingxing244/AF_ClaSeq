# Configuration Reference

Core configuration parameters for AF_ClaSeq workflows, plus a recommended end-to-end pipeline that chains the four workflows together.

[← README](../README.md) · [Workflows](divide_and_conquer.md) · [LOO](leave_one_out.md) · [M-Fold](m_fold_sampling_voting.md) · [Occurrence](occurrence_voting.md) · [UMAP](umap_voting.md) · [Config](configuration.md) · [Troubleshooting](troubleshooting.md)

## Core Configuration Parameters

### Input/Output Configuration
```yaml
input:
  a3m_file: "/path/to/input/alignment.a3m"  # Input alignment file
  fasttree_binary: "/path/to/FastTree"      # FastTree executable path

output:
  working_dir: "/path/to/output/directory"  # Working directory for outputs
```

### Clade Splitting
```yaml
clade_splitting:
  min_clade_size: 40          # Minimum sequences per clade
  max_clade_size: 150         # Maximum sequences per clade
  coverage_filter:
    enabled: true             # Enable coverage-based filtering
    threshold: 0.7            # Minimum non-gap content ratio (70%)
```

### Sequence Shuffling
```yaml
shuffling:
  num_shuffles: 10            # Number of random shuffles per clade
  group_size: 8               # Number of sequences per group
  random_seed: 42             # For reproducible shuffling
```

### ColabFold Integration
```yaml
colabfold:
  conda_env: "/path/to/colabfold/conda/env"  # ColabFold environment path
  num_models: 1                              # Number of models to generate
  num_seeds: 1                               # Number of random seeds
  max_concurrent_jobs: 80                    # Maximum parallel jobs
```

### SLURM Configuration
```yaml
slurm:
  account: "your_slurm_account"   # SLURM account name
  partition: "your_partition"     # SLURM partition name
  time: "01:00:00"                # Job time limit
  memory: "32G"                   # Memory per job
  cpus: 8                         # CPU cores per job
```

For complete parameter descriptions and advanced options, see the example configuration files in `example/config_examples/`.

## Putting It All Together: Complete Pipeline Example

Here's a recommended workflow from start to finish:

### Scenario: You have a large MSA and want to predict multiple conformational states

**Step 1: Divide-and-Conquer**
```bash
# Goal: Get initial predictions from large MSA
python scripts/run_divide_and_conquer.py --config my_dac_config.yaml
# Output: Many predictions organized by phylogenetic clades
```

**Step 2: Leave-One-Out**
```bash
# Goal: Find impactful sequences from promising clade
# Use sequences from best clade as input
python scripts/run_leave_one_out.py my_loo_config.yaml
# Output: significant_impact_sequences.a3m
```

**Step 3: Occurrence Voting**
```bash
# Goal: Select optimal subset from impactful sequences
# Use significant sequences from LOO as input
python scripts/run_occurrence_voting.py my_voting_config.yaml
# Output: final_top_16_sequences.a3m
```

**Step 4: M-Fold Sampling**
```bash
# Goal: Explore conformational space with optimal sequences
# Use top sequences from voting as input
python scripts/run_m_fold_sampling_voting.py my_mfold_config.yaml
# Output: High-quality predictions binned by conformation
```

---
[← Back to README](../README.md)
