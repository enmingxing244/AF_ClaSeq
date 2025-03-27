# AF-ClaSeq: Protein Structure Prediction Pipeline

AF-ClaSeq is a comprehensive pipeline for protein structure prediction and analysis using AlphaFold2 and sequence-based sampling approaches. The pipeline is designed to leverage sequence purification for accurate prediction of multiple conformational states.

## Overview

AF-ClaSeq uses evolutionary sequence information to generate diverse sequence profiles, which are then used to predict protein structures with AlphaFold2. The pipeline consists of multiple stages, including iterative shuffling, M-fold sampling, sequence voting, recompilation, and prediction analysis.

## Features

- **Iterative Shuffling**: Enriches sequences through iterative selection based on structure prediction metrics
- **M-fold Sampling**: Performs multiple rounds of sequence sampling to generate diverse predictions
- **Sequence Voting**: Analyzes the distribution of sequence-based predictions to identify patterns
- **Recompilation & Prediction**: Recompiles selected sequences and predicts structures
- **Analysis & Visualization**: Provides comprehensive visualization tools for analyzing results

## Installation

### Prerequisites

- Python 3.10 or newer
- SLURM compute cluster (recommended for large-scale predictions)
- AlphaFold2 installation (or ColabFold)

### Using Poetry

The recommended way to install AF-ClaSeq is using Poetry:

```bash
# Clone the repository
git clone https://github.com/enmingxing244/AF_ClaSeq.git
cd AF_ClaSeq

# Install with Poetry
poetry install

# For full installation with all dependencies
poetry install --extras "full"

# For visualization tools only
poetry install --extras "visualization"

# For analysis tools only
poetry install --extras "analysis"
```

### Using pip

Alternatively, you can install using pip:

```bash
pip install git+https://github.com/enmingxing244/AF_ClaSeq.git
```

## Demo Usage

AF-ClaSeq includes an example dataset for the KaiB protein in the `examples/KaiB` directory. Here's how to run a simple demo:

### Example with KaiB Protein

1. First, generate the default ColabFold predicted results using the provided MSA input:
```bash
pip install git+https://github.com/enmingxing244/AF_ClaSeq.git
```




## Configuration

AF-ClaSeq requires a YAML configuration file to specify pipeline parameters. Below is an example configuration:

```yaml
general:
  protein_name: "my_protein"
  base_dir: "/path/to/output"
  config_file: "/path/to/config.json"
  default_pdb: "/path/to/reference.pdb"
  source_a3m: "/path/to/starting.a3m"
  num_models: 5
  num_bins: 10
  coverage_threshold: 0.8
  random_seed: 42
  plot_initial_color: "blue"
  plot_end_color: "red"

pipeline_control:
  stages:
    - "01_ITER_SHUF_RUN"
    - "01_ITER_SHUF_ANALYSIS"
    - "02_M_FOLD_SAMPLING_RUN"
    - "02_M_FOLD_SAMPLING_PLOT"
    - "03_VOTING_RUN"
    - "04_RECOMPILE_PREDICT_RUN"
    - "05_PURE_SEQ_PLOT_RUN"
  check_interval: 60

slurm:
  conda_env_path: "/path/to/conda/env"
  slurm_account: "account_name"
  slurm_output: "slurm-%j.out"
  slurm_error: "slurm-%j.err"
  slurm_nodes: 1
  slurm_gpus_per_task: 1
  slurm_tasks: 1
  slurm_cpus_per_task: 8
  slurm_time: "24:00:00"
  slurm_partition: "gpu"
  max_workers: 4

iterative_shuffling:
  iter_shuf_input_a3m: "/path/to/input.a3m"
  seq_num_per_shuffle: 100
  num_shuffles: 10
  num_iterations: 5
  quantile: 0.8
  plddt_threshold: 70.0
  resume_from_iter: 0
  enrich_filter_criteria: ["plddt", "rmsd"]
  iter_shuf_random_select: 10
  iter_shuf_plot_num_cols: 4
  iter_shuf_plot_x_min: 0
  iter_shuf_plot_x_max: 100
  iter_shuf_plot_y_min: 0
  iter_shuf_plot_y_max: 100
  iter_shuf_plot_xticks: [0, 20, 40, 60, 80, 100]
  iter_shuf_plot_bin_step: 5
  iter_shuf_combine_threshold: 0.8

m_fold_sampling:
  m_fold_samp_input_a3m: "/path/to/input.a3m"
  rounds: 3
  m_fold_group_size: 20
  m_fold_random_select: 10
  m_fold_metric1_min: 0
  m_fold_metric1_max: 100
  m_fold_metric2_min: 0
  m_fold_metric2_max: 10
  m_fold_metric1_ticks: [0, 25, 50, 75, 100]
  m_fold_metric2_ticks: [0, 2, 4, 6, 8, 10]
  m_fold_count_min: 0
  m_fold_count_max: 100
  m_fold_log_scale: false
  m_fold_n_plot_bins: 20
  m_fold_gradient_ascending: true
  m_fold_linear_gradient: true
  m_fold_plddt_threshold: 70.0
  m_fold_figsize: [10, 6]
  m_fold_show_bin_lines: true

sequence_voting:
  vote_threshold: 5
  vote_min_value: 0
  vote_max_value: 100
  use_focused_bins: true
  vote_figsize: [12, 8]
  vote_y_min: 0
  vote_y_max: 100
  vote_x_ticks: [0, 2, 4, 6, 8, 10]

recompile_predict:
  bin_numbers_1: [1, 3, 5, 7, 9]
  bin_numbers_2: [2, 4, 6, 8, 10]
  metric_name_1: "plddt"
  metric_name_2: "rmsd"
  combine_bins: true
  prediction_num_model: 5
  prediction_num_seed: 3

pure_sequence_plotting:
  metric1_min: 0
  metric1_max: 100
  metric2_min: 0
  metric2_max: 10
  metric1_ticks: [0, 25, 50, 75, 100]
  metric2_ticks: [0, 2, 4, 6, 8, 10]
  plddt_threshold: 70.0
  figsize: [12, 10]
  dpi: 300
```

## Filter Configuration

In addition to the YAML configuration file, AF-ClaSeq requires a JSON configuration file specifying filter criteria for structure evaluation. Here's an example:

```json
{
  "filter_criteria": [
    {
      "name": "plddt",
      "type": "plddt",
      "min_value": 0,
      "max_value": 100,
      "description": "AlphaFold2 pLDDT confidence score",
      "higher_is_better": true
    },
    {
      "name": "rmsd",
      "type": "rmsd",
      "reference_pdb": "/path/to/reference.pdb",
      "min_value": 0,
      "max_value": 10,
      "description": "RMSD to reference structure",
      "higher_is_better": false
    }
  ]
}
```

## Usage

To run the AF-ClaSeq pipeline:

```bash
python -m af_claseq.run_af_claseq_pipeline config.yaml
```

Where `config.yaml` is your configuration file.

### Pipeline Stages

AF-ClaSeq consists of the following stages, which can be enabled or disabled in the configuration:

1. **01_RUN**: Iterative shuffling of sequences
2. **01_ANALYSIS**: Analysis of iterative shuffling results
3. **02_RUN**: M-fold sampling of sequences
4. **02_PLOT**: Plotting and analysis of M-fold sampling results
5. **03**: Sequence voting analysis
6. **04**: Recompilation of sequences and structure prediction
7. **05**: Plotting and analysis of prediction results

## Example Workflow

Here's a typical workflow using AF-ClaSeq:

1. **Prepare Input Files**:
   - Initial MSA file (A3M format)
   - Reference PDB structure (optional)
   - Configuration files (YAML and JSON)

2. **Run Pipeline**:
   ```bash
   python -m af_claseq.run_af_claseq_pipeline config.yaml
   ```

3. **Analyze Results**:
   - Examine plots in output directories
   - Use predicted structures for further analysis

## Output Structure

The pipeline creates a directory structure according to the configured `base_dir`:

```
base_dir/
├── logs/
│   └── af_claseq_pipeline.log
├── 01_iterative_shuffling/
│   ├── iteration_1/
│   ├── iteration_2/
│   └── ...
├── 02_m_fold_sampling/
│   ├── round_1/
│   ├── round_2/
│   ├── csv/
│   └── plot/
├── 03_voting/
│   ├── criterion_1/
│   └── criterion_2/
├── 04_recompile/
│   ├── criterion_1/
│   └── criterion_2/
└── 05_plots/
    ├── criterion_1/
    └── criterion_2/
```

## Advanced Usage

### Running on SLURM Cluster

AF-ClaSeq is designed to work with SLURM job schedulers. Configure the SLURM parameters in the configuration file to match your cluster setup.

### Custom Filter Criteria

You can define custom filter criteria in the JSON configuration file to evaluate structures based on your specific needs.

### Partial Pipeline Execution

You can run specific stages of the pipeline by modifying the `stages` list in the configuration file.

## Dependencies

AF-ClaSeq depends on the following Python packages:

- numpy
- biopython
- tqdm
- pyyaml
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- plotly
- networkx
- loguru
- h5py
- biotite
- mdanalysis
- mdtraj
- nglview
- py3dmol

Optional dependencies:
- colabfold
- alphafold-colabfold
- jax
- jaxlib
- dm-haiku
- ml-collections

## Citation

If you use AF-ClaSeq in your research, please cite:

```
@article{xing2025afclaseq,
  title={AF-ClaSeq: Leveraging Sequence Purification for Accurate Prediction of Multiple Conformational States with AlphaFold2},
  author={Xing, Enming},
  year={2025},
  publisher={GitHub},
  url={https://github.com/enmingxing244/AF_ClaSeq}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please contact:
- Enming Xing - xing244@osu.edu

## Acknowledgments

- AlphaFold2 team for their groundbreaking protein structure prediction method
- Contributors to the open-source bioinformatics tools used in this project