# Column Shuffle Pipeline

An integrated pipeline for contact-based MSA column shuffling to analyze state-specific residue coupling in protein conformational transitions.

## Overview

This pipeline performs the following steps:

1. **Contact Map Construction**: Constructs CB-based contact maps from two PDB structures representing different conformational states, then identifies state-specific unique residue pairs
2. **Column Shuffling**: Shuffles MSA columns at unique pair positions for each state
3. **Parallel Structure Prediction**: Submits ColabFold jobs for all shuffled MSAs in parallel
4. **Structure Analysis**: Calculates user-defined structural metrics on predicted structures
5. **Plotting**: Generates publication-quality 2D scatter plots
6. **Show Positions**: Extracts and displays aligned residues at unique pair positions from the original MSA

## Installation

No additional installation required. The pipeline uses existing AF_ClaSeq utilities and dependencies.

## Files

- `column_shuffle_pipeline.py` - Main pipeline script
- `config.py` - Configuration dataclasses
- `example_config.yaml` - Template configuration file
- `README.md` - This file

## Usage

### 1. Prepare Your Configuration File

Copy and customize `example_config.yaml`:

```bash
cp scripts/column_shuf/example_config.yaml my_config.yaml
```

Edit `my_config.yaml` to specify:
- Protein name
- Input PDB files for both states
- Input MSA files for both states
- Structure metrics config file (JSON)
- SLURM parameters
- Plotting parameters

### 2. Run the Pipeline

Run all stages:
```bash
python scripts/column_shuf/column_shuffle_pipeline.py my_config.yaml
```

Run specific stages only:
```bash
python scripts/column_shuf/column_shuffle_pipeline.py my_config.yaml --stages contact_maps shuffle
```

Show aligned residues at unique pair positions:
```bash
python scripts/column_shuf/column_shuffle_pipeline.py my_config.yaml --stages show_positions
```

Dry run (validate config without executing):
```bash
python scripts/column_shuf/column_shuffle_pipeline.py my_config.yaml --dry-run
```

### 3. Check Results

Results will be organized in the base directory specified in your config:

```
base_dir/
├── logs/
│   └── column_shuffle_pipeline.log
├── 01_contact_maps/
│   ├── state1_contact_map.npy
│   ├── state2_contact_map.npy
│   ├── difference_map.npy
│   ├── unique_pairs.json
│   ├── {state1_name}_contact_map.png
│   ├── {state2_name}_contact_map.png
│   └── difference_map.png
├── 02_shuffled_msas/
│   ├── {state1_name}/
│   │   ├── {state1_name}_shuffle_v01.a3m
│   │   └── ... (up to vN)
│   └── {state2_name}/
│       └── ...
├── 03_predictions/
│   ├── {state1_name}/
│   │   ├── shuffle_v01/ (ColabFold outputs)
│   │   └── ...
│   └── {state2_name}/
│       └── ...
├── 04_analysis/
│   ├── {state1_name}/
│   │   └── structure_analysis.csv
│   └── {state2_name}/
│       └── structure_analysis.csv
├── 05_plots/
│   ├── {state1_name}/
│   │   ├── scatter_plot.png
│   │   ├── scatter_plot.pdf
│   │   └── scatter_plot.svg
│   └── {state2_name}/
│       ├── scatter_plot.png
│       ├── scatter_plot.pdf
│       └── scatter_plot.svg
└── 06_position_alignments/
    ├── {state1_name}_unique_positions.txt
    └── {state2_name}_unique_positions.txt
```

## Configuration Parameters

### General Section

- `protein_name`: Name of the protein (used for output naming)
- `base_dir`: Base output directory for all results
- `config_file`: Path to JSON file defining structural metrics
- `random_seed`: Random seed for reproducibility
- `use_composite_metrics`: Enable composite metrics calculation (default: false)
- `state1_pdb`, `state1_a3m`, `state1_name`: State 1 inputs
- `state2_pdb`, `state2_a3m`, `state2_name`: State 2 inputs
- `num_shuffles`: Number of shuffled MSA replicates per state (default: 20)
- `contact_threshold`: Threshold for identifying unique pairs (default: 0.8)
- `sigmoid_center`: Sigmoid center distance in Å (default: 8.0)
- `sigmoid_steepness`: Sigmoid steepness (default: 1.0)
- `min_sequence_separation`: Minimum residue separation for pairs (default: 10)
- `unique_pair_residue_range`: Optional [start, end] range to filter unique pairs (null = no filtering)

### SLURM Section

- `conda_env_path`: Path to ColabFold conda environment
- `slurm_account`: SLURM account name
- `slurm_partition`: SLURM partition (default: "nextgen")
- `slurm_time`: Time limit per job (default: "01:00:00")
- `slurm_nodes`, `slurm_gpus_per_task`, `slurm_cpus_per_task`: Resource allocation
- `max_workers`: Maximum concurrent jobs (default: 200)
- `num_models`: Number of models per prediction (default: 1)

### Plotting Section

- `metric1_name`: X-axis metric name (must match column in structure analysis CSV)
- `metric2_name`: Y-axis metric name
- `metric1_color`, `metric2_color`: Color gradients [start, end]
- `x_lim`, `y_lim`: Axis limits [min, max]

## Structure Metrics Config File

The pipeline requires a JSON config file defining structural metrics to calculate. Example format:

```json
{
  "basics": {
    "full_index": {"start": 1, "end": 91}
  },
  "filter_criteria": [
    {
      "name": "state1_tmscore",
      "type": "tmscore",
      "method": "above",
      "ref_pdb": "/path/to/state1_reference.pdb"
    },
    {
      "name": "state2_tmscore",
      "type": "tmscore",
      "method": "above",
      "ref_pdb": "/path/to/state2_reference.pdb"
    }
  ]
}
```

## Column Shuffling Methodology

The column shuffling approach:

1. **Query sequence remains FIXED** at all positions
2. **Non-query sequences are shuffled** at specified column positions
3. Each position shuffles **independently** across all non-query sequences
4. Multiple shuffled replicates are generated for statistical analysis

For example, if positions [1, 2, 3] are selected:
- Query sequence: unchanged
- Column 1: residues from all non-query sequences → randomly permuted
- Column 2: residues from all non-query sequences → randomly permuted
- Column 3: residues from all non-query sequences → randomly permuted
- Process repeated N times to create N independent replicates

## Contact Map Construction

Contact maps are constructed using:
- CB atoms for distance calculation (CA for glycine)
- Pairwise Euclidean distances
- Sigmoid normalization: `1.0 / (1.0 + exp(steepness * (distance - center)))`
- Difference map identifies state-specific contacts

## Example: KaiB Protein

```yaml
general:
  protein_name: "KaiB"
  base_dir: "/fs/ess/PAA0203/xing244/AF_ClaSeq/results_updated/KaiB/column_shuffle_test"
  config_file: "/fs/ess/PAA0203/xing244/AF_ClaSeq/results_updated/KaiB/configs/config_2qke_5jyt_tmscore.json"

  state1_pdb: "/path/to/2qkeE.pdb"
  state1_a3m: "/path/to/2qke_sampling.a3m"
  state1_name: "fold_switched"

  state2_pdb: "/path/to/5jytA.pdb"
  state2_a3m: "/path/to/5jyt_sampling.a3m"
  state2_name: "ground_state"

  num_shuffles: 20
  contact_threshold: 0.8
```

## Troubleshooting

**Import Error for MSAColumnShuffler:**
- Ensure the KaiB example code exists at the expected path
- The pipeline looks for: `results_updated/KaiB/run1/column_shuffle/02_column_shuffle/column_shuffle_utility.py`

**SLURM Job Failures:**
- Check SLURM logs in job directories
- Verify conda environment path is correct
- Ensure sufficient time limit for predictions

**Structure Analysis Errors:**
- Verify structure metrics config file exists and is valid JSON
- Check that metric names match between config and plotting section

## References

Based on the column shuffling methodology from:
`/fs/ess/PAA0203/xing244/AF_ClaSeq/results_updated/KaiB/run1/column_shuffle/02_column_shuffle/`

## Support

For issues or questions, refer to the implementation plan at:
`/users/PAA0203/xing244/.claude/plans/stateful-coalescing-crab.md`
