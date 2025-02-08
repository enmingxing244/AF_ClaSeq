# AF_ClaSeq: AlphaFold-based Conformational Landscape Sequence Analysis

AF_ClaSeq is a comprehensive pipeline for analyzing protein conformational landscapes using AlphaFold2 predictions and sequence-structure relationships. This tool enables the identification and classification of sequences based on their structural preferences through iterative sampling and voting mechanisms.

## Features

- Iterative sequence shuffling and structure prediction
- M-fold sampling for conformational space exploration
- Multi-dimensional sequence voting system (1D/2D/3D)
- Structure quality assessment and filtering
- Advanced visualization and analysis tools
- Parallel processing support with SLURM integration

## Installation

```bash
# Clone the repository
git clone https://github.com/enmingxing244/AF_ClaSeq.git
cd AF_ClaSeq

# Create and activate conda environment
conda create -n af_claseq python=3.10
conda activate af_claseq

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- Python 3.10+
- Colabfold
- BioPython
- NumPy
- Pandas
- Matplotlib
- TMalign

## Directory Structure

```
AF_ClaSeq/
├── scripts/                 # Main execution scripts
│   ├── 01_iterative_shuffling_run.py
│   ├── 02_M_fold_sampling_run.py
│   ├── 03_sequence_voting_run.py
│   └── 04_recompile_run.py
├── src/af_claseq/          # Core functionality modules
│   ├── sequence_processing.py
│   ├── structure_analysis.py
│   ├── voting.py
│   └── slurm_utils.py
└── docs/                   # Documentation
```

## Detailed Usage Guide

### 1. Iterative Sequence Shuffling

The first step in the pipeline performs iterative sequence filtering and structure prediction to identify promising sequence subsets.

```bash
python scripts/01_iterative_shuffling_run.py \
    --input_a3m input.a3m \                    # Input multiple sequence alignment in A3M format
    --default_pdb reference.pdb \              # Reference PDB structure
    --base_dir output_dir \                    # Output directory for results
    --config_file config.json \                # Configuration file with filter criteria
    --coverage_threshold 0.8 \                 # Minimum sequence coverage (0-1)
    --num_iterations 3 \                       # Number of iteration rounds
    --group_size 10 \                          # Number of sequences per group
    --num_shuffles 10 \                        # Number of shuffle attempts per iteration
    --quantile 0.6 \                          # Quantile threshold for filtering
    --plddt_threshold 75                       # Minimum pLDDT score threshold
```

Optional SLURM-specific arguments:
```bash
    --conda_env_path /path/to/conda/env \     # Path to conda environment
    --slurm_account ACCOUNT \                 # SLURM account name
    --slurm_time "04:00:00" \                # Wall time limit
    --slurm_nodes 1 \                        # Number of nodes per job
    --slurm_gpus_per_task 1 \                # GPUs per task
    --max_workers 64                         # Maximum concurrent workers
```

### 2. M-fold Sampling

This step systematically explores the conformational space through M-fold cross-validation sampling.

```bash
python scripts/02_M_fold_sampling_run.py \
    --input_a3m filtered_sequences.a3m \       # Input MSA from previous step
    --default_pdb reference.pdb \              # Reference PDB structure
    --base_dir output_dir \                    # Output directory
    --group_size 10 \                          # Sequences per group
    --coverage_threshold 0.0 \                 # Additional coverage filter (0 = no filtering)
    --random_select_num_seqs 1000 \           # Number of sequences to randomly select
    --max_workers 64                          # Maximum concurrent workers
```

The M-fold sampling creates:
- Initial random splits
- Systematic sampling combinations
- Structure predictions for each split
- Quality assessment metrics

### 3. Sequence Voting

The voting system classifies sequences based on their structural preferences using multiple metrics.

```bash
python scripts/03_sequence_voting_run.py \
    --sampling_dir sampling_output \           # Directory containing sampling results
    --source_msa input.a3m \                  # Original MSA file
    --config_path config.json \               # Configuration file
    --num_bins 20 \                           # Number of bins for classification
    --vote_threshold 0.2 \                    # Minimum vote ratio threshold
    --max_workers 88 \                        # Maximum concurrent workers
    --output_dir voting_output \              # Output directory
    --min_value 0.5 \                        # Minimum value for metric range
    --max_value 1.0 \                        # Maximum value for metric range
    --initial_color "#d3b0b0" \              # Color for plotting
    --use_focused_bins \                     # Use focused binning with outliers
    --plddt_threshold 70                     # pLDDT score threshold
```

#### Voting Dimensions
1. **1D Voting**: Single metric (e.g., TM-score)
   ```json
   {
       "filter_criteria": [{
           "name": "tmscore",
           "type": "tmscore",
           "method": "above"
       }]
   }
   ```

2. **2D Voting**: Two metrics (e.g., TM-score and pLDDT)
   ```json
   {
       "filter_criteria": [
           {
               "name": "tmscore",
               "type": "tmscore",
               "method": "above"
           },
           {
               "name": "plddt",
               "type": "plddt",
               "method": "above"
           }
       ]
   }
   ```

3. **3D Voting**: Three metrics
   ```json
   {
       "filter_criteria": [
           {
               "name": "tmscore",
               "type": "tmscore",
               "method": "above"
           },
           {
               "name": "plddt",
               "type": "plddt",
               "method": "above"
           },
           {
               "name": "rmsd",
               "type": "rmsd",
               "method": "below"
           }
       ]
   }
   ```

### 4. Sequence Recompilation

Final step to select and validate sequences from specific bins:

```bash
python scripts/04_recompile_run.py \
    --output_dir final_output \               # Output directory
    --bin_numbers 4 5 6 \                     # Bin numbers to compile from
    --source_msa input.a3m \                  # Source MSA file
    --num_selections 20 \                     # Number of random selections
    --default_pdb reference.pdb \             # Reference PDB file
    --seq_num_per_selection 12 \              # Sequences per selection
    --voting_results_path results.csv \       # Voting results file
    --raw_votes_json votes.json \             # Raw votes data
    --initial_color "#2486b9" \              # Plot color
    --num_total_bins 20 \                    # Total number of bins
    --combine_bins                           # Combine sequences from all bins
```

#### Output Structure
```
final_output/
├── prediction/                # Main prediction results
│   └── bin_*/
│       └── sequences.a3m
├── control_prediction/        # Control group predictions
│   └── bin_*/
│       └── random_seq.a3m
├── random_selection/         # Random subsets for validation
│   └── bin_*/
│       └── selection_*.a3m
└── plots/                    # Visualization outputs
    └── *_sequence_vote_ratios.png
```

### Configuration File Details

The `config.json` file supports various metric types and filtering methods:

```json
{
    "filter_criteria": [
        {
            "name": "tmscore",
            "type": "tmscore",
            "method": "above",
            "ref_pdb": "reference.pdb"
        },
        {
            "name": "distance",
            "type": "distance",
            "method": "below",
            "indices": {
                "set1": [10, 11, 12],
                "set2": [20, 21, 22]
            }
        },
        {
            "name": "angle",
            "type": "angle",
            "method": "above",
            "indices": {
                "domain1": [1, 2, 3],
                "domain2": [30, 31, 32],
                "hinge": [15, 16, 17]
            }
        },
        {
            "name": "rmsd",
            "type": "rmsd",
            "method": "below",
            "superposition_indices": {
                "start": 1,
                "end": 50
            },
            "rmsd_indices": {
                "start": 51,
                "end": 100
            }
        }
    ],
    "basics": {
        "full_index": {
            "start": 1,
            "end": 100
        },
        "local_index": {
            "start": 20,
            "end": 80
        }
    }
}
```

### Error Handling and Troubleshooting

1. **Common Issues**:
   - Missing dependencies: Ensure all required packages are installed
   - GPU memory errors: Reduce batch size or group size
   - SLURM job failures: Check resource requirements
   - File permissions: Ensure write access to output directories

2. **Quality Control**:
   - Monitor pLDDT scores for structure quality
   - Check vote distribution plots for anomalies
   - Validate results against control groups
   - Verify sequence coverage and representation

3. **Performance Optimization**:
   - Adjust `max_workers` based on system resources
   - Use appropriate GPU allocation for structure prediction
   - Monitor memory usage with large sequence sets
   - Consider using precomputed metrics when available

## Output Structure

```
output_dir/
├── 01_iterative_shuffling/
│   └── Iteration_*/
├── 02_sampling/
│   └── sampling_*/
├── 03_voting/
│   ├── voting_results.csv
│   └── vote_distribution.png
└── 04_recompile/
    ├── prediction/
    ├── control_prediction/
    └── plots/
```

## Visualization

The pipeline generates various visualization outputs:
- Sequence vote distribution plots
- Structure quality metric distributions
- Conformational landscape maps
- Validation and control comparisons

## Performance Considerations

- Memory usage scales with sequence count and group size
- GPU requirements depend on AlphaFold2 configuration
- Parallel processing capability through SLURM integration
- Recommended minimum 32GB RAM for standard datasets

## Error Handling

The pipeline includes comprehensive error handling:
- Input validation
- File existence checks
- Structure quality assessment
- Vote confidence thresholds

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use AF_ClaSeq in your research, please cite:
[Citation information to be added]

## Contact

For questions and support, please open an issue on GitHub or contact [your contact information].
