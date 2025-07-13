# Migration Guide: From Iterative Shuffling to Hit Expand

## Overview

**UPDATE: As of v1.1.1, the iterative_shuffling module has been completely removed. This guide is kept for historical reference.**

This guide documents the migration from the deprecated `iterative_shuffling` step to the new `hit_expand` functionality, which integrates MSA pipeline capabilities with MMseqs2 clustering.

## What Changed

### New Functionality
- **MMseqs2 Clustering**: Automatic sequence clustering with configurable parameters
- **MSA Pipeline Integration**: Full integration of MSA pipeline structure prediction and analysis
- **Hit Expansion**: BLOSUM62-based similarity search for expanding hit sequences
- **Enhanced Structure Analysis**: Improved structure analysis with configurable thresholds
- **Unified Output Management**: Better organized output directory structure

### Deprecated Functionality
- `01_iterative_shuffling` module (removed in v1.1.1)
- `IterShufEnrichRunner`, `IterShufEnrichPlotter`, `IterShufEnrichCombiner` classes

## Configuration Changes

### Old Configuration (Deprecated)
```yaml
pipeline_control:
  stages:
    - "01_ITER_SHUF_RUN"
    - "01_ITER_SHUF_ANALYSIS"
    # ... other stages

iterative_shuffling:
  iter_shuf_input_a3m: "/path/to/input.a3m"
  num_iterations: 8
  num_shuffles: 10
  # ... other parameters
```

### New Configuration (Recommended)
```yaml
pipeline_control:
  stages:
    - "01_HIT_EXPAND_RUN"           # New stage
    - "01_HIT_EXPAND_ANALYSIS"      # New stage
    # ... other stages

hit_expand:
  input_msa: "/path/to/input.a3m"
  mmseqs_bin: "mmseqs"
  mmseqs_coverage: 0.8
  mmseqs_min_seq_id: 0.7
  num_subsets: 2000
  num_batches: 80
  # ... other parameters
```

## Directory Structure Changes

### Old Structure
```
output_dir/
├── 01_iterative_shuffling/
│   ├── iteration_1/
│   ├── iteration_2/
│   └── gathered_seq_after_iter_shuffling.a3m
├── 02_m_fold_sampling/
└── ...
```

### New Structure
```
output_dir/
├── 01_hit_expand/
│   ├── 00_clustering/
│   │   ├── cluster_result_rep_seq.fasta
│   │   └── clustered_representatives.a3m
│   ├── 01_msa_pipeline/
│   │   ├── batches/
│   │   ├── hit_expansion/
│   │   └── final_optimized_msa.a3m
│   ├── plots/
│   ├── hit_expand_final_msa.a3m
│   └── hit_expand_summary.json
├── 02_m_fold_sampling/
└── ...
```

## Migration Steps

### 1. Update Configuration File

Replace your existing configuration with the new hit_expand format:

```bash
# Copy the example configuration
cp examples/hit_expand_config.yaml your_config.yaml
```

Edit `your_config.yaml` to match your specific requirements.

### 2. Install MMseqs2 (if not already installed)

```bash
# Install MMseqs2
conda install -c conda-forge mmseqs2
# OR
wget https://github.com/soedinglab/MMseqs2/releases/latest/download/mmseqs-linux-avx2.tar.gz
tar xzf mmseqs-linux-avx2.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH
```

### 3. Update Input Paths

Update your configuration to point to the correct input files:

```yaml
hit_expand:
  input_msa: "/path/to/your/source_msa.a3m"  # Your input MSA file
```

### 4. Configure MMseqs2 Parameters

Adjust clustering parameters based on your needs:

```yaml
hit_expand:
  mmseqs_coverage: 0.8      # Coverage threshold (0.0-1.0)
  mmseqs_min_seq_id: 0.7    # Minimum sequence identity (0.0-1.0)
  mmseqs_threads: 8         # Number of threads
```

### 5. Run the Pipeline

```bash
python run_af_claseq_pipeline.py your_config.yaml
```

## Parameter Mapping

| Old Parameter | New Parameter | Notes |
|---------------|---------------|--------|
| `iter_shuf_input_a3m` | `input_msa` | Same functionality |
| `num_iterations` | `num_subsets` | Controls number of subsets instead of iterations |
| `num_shuffles` | `num_batches` | Controls batch organization |
| `seq_num_per_shuffle` | `num_random_sequences` | Sequences per subset |
| `plddt_threshold` | `plddt_threshold` | Same functionality |
| N/A | `mmseqs_coverage` | New clustering parameter |
| N/A | `mmseqs_min_seq_id` | New clustering parameter |
| N/A | `similarity_threshold` | New hit expansion parameter |

## Output File Mapping

| Old Output | New Output | Notes |
|------------|------------|--------|
| `gathered_seq_after_iter_shuffling.a3m` | `hit_expand_final_msa.a3m` | Final MSA output |
| `iteration_*/plots/` | `plots/` | Consolidated plotting |
| N/A | `00_clustering/` | New clustering results |
| N/A | `01_msa_pipeline/` | New MSA pipeline outputs |

## Backward Compatibility

The new pipeline maintains backward compatibility:

1. **Legacy Stages**: Old stage names (`01_ITER_SHUF_RUN`, `01_ITER_SHUF_ANALYSIS`) still work
2. **Legacy Configuration**: Old `iterative_shuffling` configuration is no longer supported
3. **Automatic Fallback**: Downstream stages automatically detect and use legacy outputs

## Performance Considerations

### Memory Usage
- MMseqs2 clustering may require more memory for large MSAs
- Adjust `mmseqs_threads` based on available CPU cores
- Use `mmseqs_tmp_dir` to specify a location with sufficient disk space

### Runtime
- Hit expand is generally faster than iterative shuffling for large datasets
- Clustering time depends on MSA size and similarity thresholds
- Structure prediction time is similar to the old pipeline

## Troubleshooting

### Common Issues

1. **MMseqs2 not found**
   ```
   Solution: Ensure MMseqs2 is installed and `mmseqs_bin` points to the correct executable
   ```

2. **Clustering fails with large MSAs**
   ```
   Solution: Increase memory allocation or reduce `mmseqs_threads`
   ```

3. **No hit structures found**
   ```
   Solution: Lower `plddt_threshold` or `filter_criteria_threshold`
   ```

4. **Missing input MSA**
   ```
   Solution: Verify `input_msa` path is correct and file exists
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```yaml
# Add to your configuration
hit_expand:
  monitor_jobs: true
  job_check_interval: 30.0
```

## Advanced Configuration

### Custom MMseqs2 Parameters

```yaml
hit_expand:
  mmseqs_coverage: 0.9          # Higher coverage requirement
  mmseqs_min_seq_id: 0.8        # Higher sequence identity
  mmseqs_cov_mode: 1            # Coverage of target only
  mmseqs_cluster_mode: 2        # Greedy clustering by length
```

### Structure Analysis Tuning

```yaml
hit_expand:
  plddt_threshold: 80.0         # Higher quality requirement
  filter_criteria_threshold: 0.9  # Stricter filtering
  similarity_threshold: 0.8     # Higher similarity for expansion
```

### Performance Optimization

```yaml
hit_expand:
  num_subsets: 1000            # Fewer subsets for faster execution
  num_batches: 40              # Fewer batches
  max_workers: 16              # More parallel workers
  skip_clustering: true        # Skip clustering if already done
```

## Support

For questions or issues with the migration:

1. Check the example configuration in `examples/hit_expand_config.yaml`
2. Review the implementation in `src/af_claseq/hit_expand/`
3. Create an issue in the repository with your configuration and error logs

## Deprecation Complete

As of v1.1.1, the `iterative_shuffling` functionality has been completely removed from the codebase. All users must now use `hit_expand` for sequence optimization.