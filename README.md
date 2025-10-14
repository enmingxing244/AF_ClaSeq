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
phylogenetic:
  fasttree_binary: "/path/to/FastTree"

colabfold:
  conda_env: "/path/to/colabfold/env"
```

## 📖 Usage

### 1. Divide-and-Conquer Workflow

The divide-and-conquer approach splits large sequence alignments into phylogenetically coherent clades for targeted structure prediction.

#### Configuration Example
```yaml
# Divide-and-conquer configuration
phylogenetic:
  fasttree_binary: "/path/to/FastTree"

clade_splitting:
  min_clade_size: 40
  max_clade_size: 150
  coverage_filter:
    enabled: true
    threshold: 0.7  # 70% non-gap content minimum

shuffle:
  num_shuffles: 10
  group_size: 15

colabfold:
  conda_env: "/path/to/colabfold/env"
  num_models: 1
  num_seeds: 1
  max_concurrent_jobs: 90

slurm:
  account: "your_account"
  partition: "gpu_partition"
  time: "02:30:00"
  memory: "32G"
  cpus: 8
```

#### Running Divide-and-Conquer
```bash
python scripts/run_divide_and_conquer.py \
  --config config/divide_and_conquer_config.yaml \
  --input_a3m data/protein.a3m \
  --output_dir results/protein_dac/
```

#### Workflow Steps
1. **Preprocessing**: Sequence cleaning, header normalization, coverage filtering
2. **Phylogenetic Analysis**: FastTree construction and distance-guided clade detection
3. **Sequence Shuffling**: Multiple random groupings within each clade
4. **Structure Prediction**: Concurrent ColabFold jobs via SLURM
5. **Analysis**: Statistical evaluation and visualization

### 2. Leave-One-Out Validation

Cross-validation framework for assessing prediction robustness by systematically excluding sequences.

```bash
python scripts/run_leave_one_out.py \
  --config config/leave_one_out_config.yaml \
  --input_a3m data/protein.a3m \
  --target_sequence_id "target_seq_001"
```

### 3. M-Fold Sampling Workflow

Statistical sampling approach for generating diverse structure predictions.

```bash
python scripts/run_m_fold_sampling_voting.py \
  --config config/unified_af_claseq_config.yaml \
  --stage "02_M_FOLD_SAMPLING_RUN"
```

### 4. Occurrence Voting

Consensus-based sequence selection using occurrence frequency analysis.

```bash
python scripts/run_occurrence_voting.py \
  --config config/occurrence_voting_config.yaml \
  --input_dir results/predictions/ \
  --output_dir results/voting/
```

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

#### Phylogenetic Analysis
```yaml
phylogenetic:
  fasttree_binary: "/usr/local/bin/FastTree"  # FastTree executable path

clade_splitting:
  min_clade_size: 40          # Minimum sequences per clade
  max_clade_size: 150         # Maximum sequences per clade
  coverage_filter:
    enabled: true             # Enable coverage-based filtering
    threshold: 0.7            # Minimum non-gap content ratio (70%)
```

#### Sequence Processing
```yaml
shuffle:
  num_shuffles: 10            # Number of random shuffles per clade
  group_size: 15              # Target sequences per group
  random_seed: 42             # For reproducible shuffling
```

#### ColabFold Integration
```yaml
colabfold:
  conda_env: "/path/to/colabfold"  # ColabFold environment path
  num_models: 1                    # Number of models to generate
  num_seeds: 1                     # Number of random seeds
  max_concurrent_jobs: 90          # Maximum parallel jobs
```

#### SLURM Configuration
```yaml
slurm:
  account: "research_account"      # SLURM account name
  partition: "gpu_nodes"           # GPU partition name
  time: "02:30:00"                # Job time limit
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
