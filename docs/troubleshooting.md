# Tips, Best Practices & Troubleshooting

General guidance for running any workflow plus solutions to common errors and performance-tuning advice.

[← README](../README.md) · [Workflows](divide_and_conquer.md) · [LOO](leave_one_out.md) · [M-Fold](m_fold_sampling_voting.md) · [Occurrence](occurrence_voting.md) · [UMAP](umap_voting.md) · [Config](configuration.md) · [Troubleshooting](troubleshooting.md)

## General Tips and Best Practices

### Before Starting Any Workflow

1. **Test with dry-run**: Always validate your config first
2. **Start small**: Use a subset of sequences for initial testing
3. **Monitor jobs**: Use `squeue -u $USER` and check log files
4. **Check outputs**: Look at intermediate results before proceeding

### Resource Management

```yaml
# For small proteins (<200 residues)
slurm:
  time: "00:30:00"
  cpus: 4

# For medium proteins (200-400 residues)
slurm:
  time: "01:00:00"
  cpus: 8

# For large proteins (>400 residues)
slurm:
  time: "02:00:00"
  cpus: 12
```

### When Jobs Fail

1. **Check the log file** first: `tail -100 base_dir/logs/*.log`
2. **Check SLURM output**: `cat slurm-*.out`
3. **Common fixes**:
   - Increase `slurm.time` if jobs timed out
   - Reduce `max_concurrent_jobs` if queue is full
   - Check ColabFold environment is activated

### Saving Compute Time

- Use `--dry-run` and `--validate-only` modes liberally
- Start with small `num_shuffles`, `rounds`, `num_groups` for testing
- Use `--analysis-only` or `--step-only` to rerun analysis without predictions
- Keep successful predictions - use resume capabilities

For complete parameter descriptions and advanced options, see example configuration files in `example/config_examples/`.

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

---
[← Back to README](../README.md)
