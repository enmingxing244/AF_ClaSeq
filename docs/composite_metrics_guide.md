# Composite Metrics Guide for AF-ClaSeq

## Overview

The AF-ClaSeq pipeline now supports composite metrics, allowing you to create weighted combinations of multiple structural metrics for more sophisticated analysis, voting, and plotting.

## Configuration Format

To use composite metrics, add a `composite_metrics` section to your configuration JSON file:

```json
{
  "basics": {
    // ... existing basics configuration ...
  },
  "filter_criteria": [
    // ... existing filter criteria ...
  ],
  "composite_metrics": [
    {
      "name": "weighted_metric_name",
      "components": [
        {"metric": "metric1_name", "weight": 0.7},
        {"metric": "metric2_name", "weight": 0.3}
      ]
    }
  ]
}
```

## Example Configuration

Here's a complete example using ABL1 kinase structures:

```json
{
  "basics": {
    "full_index": {"start": 1, "end": 278},
    "local_index": [
      {"start": 46, "end": 55},
      {"start": 144, "end": 150}
    ]
  },
  "filter_criteria": [
    {
      "name": "6xrg_rmsd",
      "type": "rmsd",
      "method": "below",
      "superposition_indices": {"start": 1, "end": 278},
      "rmsd_indices": [{"start": 46, "end": 55}, {"start": 144, "end": 150}],
      "ref_pdb": "/path/to/6XRG_inactive.pdb"
    },
    {
      "name": "6xrg_dfg_rmsd",
      "type": "rmsd",
      "method": "below",
      "superposition_indices": {"start": 1, "end": 278},
      "rmsd_indices": [{"start": 381, "end": 383}],
      "ref_pdb": "/path/to/6XRG_inactive.pdb"
    }
  ],
  "composite_metrics": [
    {
      "name": "weighted_6xrg_composite",
      "components": [
        {"metric": "6xrg_rmsd", "weight": 0.3},
        {"metric": "6xrg_dfg_rmsd", "weight": 0.7}
      ]
    }
  ]
}
```

## Key Features

### 1. Multiple Components
You can combine any number of metrics:
```json
{
  "name": "multi_metric_composite",
  "components": [
    {"metric": "rmsd1", "weight": 0.4},
    {"metric": "rmsd2", "weight": 0.3},
    {"metric": "angle1", "weight": 0.2},
    {"metric": "distance1", "weight": 0.1}
  ]
}
```

### 2. Automatic Weight Normalization
If weights don't sum to 1.0, they will be automatically normalized:
```json
{
  "name": "auto_normalized",
  "components": [
    {"metric": "metric1", "weight": 2},
    {"metric": "metric2", "weight": 3}
  ]
}
// Effective weights: metric1=0.4, metric2=0.6
```

### 3. Use in Analysis and Plotting
Composite metrics can be used just like regular metrics:
- In 1D/2D scatter plots
- For sequence voting
- In structure filtering

## Usage in Pipeline Stages

### Hit Expand
Composite metrics are automatically calculated during structure analysis:
```python
# The pipeline automatically extracts composite metrics from config
composite_metrics = filter_config.get("composite_metrics", [])
```

### M-fold Sampling Plots
Use composite metrics in plotting by specifying their names:
```python
plot_m_fold_sampling_1d(
    metric_name="weighted_6xrg_composite",  # Use composite metric name
    # ... other parameters ...
)
```

### Sequence Voting
Composite metrics can be used as voting criteria:
```bash
# Use composite metric for voting
python run_sequence_voting.py \
    --filter-criterion "weighted_6xrg_composite" \
    # ... other parameters ...
```

## Best Practices

1. **Meaningful Weights**: Choose weights that reflect the biological importance of each component
2. **Component Availability**: Ensure all component metrics are defined in `filter_criteria`
3. **Naming Convention**: Use descriptive names for composite metrics
4. **Testing**: Validate composite metrics produce expected results before large-scale runs

## Backward Compatibility

The composite metrics feature is fully backward compatible:
- Existing configurations without `composite_metrics` work unchanged
- All existing functionality is preserved
- Composite metrics are optional and additive

## Troubleshooting

### Missing Component Metrics
If a component metric is not found, the composite metric will not be calculated and a warning will be logged.

### Invalid Values
If any component metric has a NaN or None value, the composite metric will not be calculated for that structure.

### Performance
Composite metrics add minimal computational overhead as they are calculated alongside regular metrics.