# Scientific Background & Output Analysis

The methodological rationale behind AF_ClaSeq's core strategies, plus the structure and meaning of the results it produces.

[← README](../README.md) · [Workflows](divide_and_conquer.md) · [LOO](leave_one_out.md) · [M-Fold](m_fold_sampling_voting.md) · [Occurrence](occurrence_voting.md) · [UMAP](umap_voting.md) · [Config](configuration.md) · [Troubleshooting](troubleshooting.md)

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

---
[← Back to README](../README.md)
