# Hit Expand Plotting Enhancements

## Overview

The enhanced plotting module (`plotting_enhanced.py`) incorporates sophisticated visualization capabilities inspired by the MSA pipeline, providing comprehensive analysis and reporting for the hit expand process.

## Key Features

### 1. **RMSD Scatter Plots with pLDDT Coloring**
- Color-coded scatter plots showing RMSD values colored by pLDDT scores
- Automatic threshold visualization with red dashed lines
- Mean and median value overlays for quick statistical assessment
- Support for multiple RMSD metrics from AF-ClaSeq config

### 2. **Quality Distribution Plots with KDE**
- Histogram distributions with kernel density estimation overlays
- Multi-panel layouts showing:
  - pLDDT distribution with threshold line
  - RMSD distributions for multiple metrics
  - Quality category pie charts
  - Normalized box plots for all metrics

### 3. **2x2 Quality Summary Plot (MSA Pipeline Signature)**
A comprehensive 4-panel summary including:
- **Panel 1**: pLDDT vs RMSD scatter plot
- **Panel 2**: Quality category bar chart (Very Low/Low/Confident/Very High)
- **Panel 3**: pLDDT histogram with mean/median lines
- **Panel 4**: Summary statistics table

### 4. **Metric Correlation Heatmap**
- Correlation analysis between all numeric metrics
- Triangular heatmap with correlation coefficients
- Color-coded from -1 (negative correlation) to +1 (positive correlation)

### 5. **Pipeline Progress Visualization**
- Structure count evolution through pipeline stages
- Quality metrics evolution plots
- Side-by-side comparison of total vs high-quality structures

### 6. **MSA Evolution Analysis**
- Sequence count evolution through pipeline stages
- Sequence length distribution comparisons
- Sequence diversity analysis (unique sequence ratios)
- Gap content evolution tracking

### 7. **Comprehensive HTML Report**
- Interactive HTML report with all plots embedded
- Summary statistics table
- Organized plot sections
- Timestamp and metadata

## Usage

### Basic Usage

```python
from af_claseq.hit_expand.plotting_enhanced import EnhancedHitExpandPlotter
from af_claseq.hit_expand.config import HitExpandPlottingConfig

# Initialize plotter
plotter = EnhancedHitExpandPlotter(
    base_dir=Path("path/to/hit_expand/output"),
    logger=logger
)

# Create plotting configuration
plot_config = HitExpandPlottingConfig(
    figsize=(14, 8),
    dpi=300,
    plddt_threshold=70.0,
    filter_criteria_threshold=0.8
)

# Generate comprehensive plots
saved_plots = plotter.create_comprehensive_analysis_plots(
    msa_output=Path("path/to/final_msa.a3m"),
    config_file="path/to/af_claseq_config.json",
    plots_dir=Path("path/to/output/plots"),
    plot_config=plot_config
)
```

### Integration with Pipeline

The enhanced plotter is automatically used in the pipeline when available:

```python
def analyze_hit_expand(self) -> bool:
    try:
        from af_claseq.hit_expand.plotting_enhanced import EnhancedHitExpandPlotter
        use_enhanced = True
    except ImportError:
        from af_claseq.hit_expand.plotting import HitExpandPlotter
        use_enhanced = False
```

## Plot Types Generated

1. **rmsd_scatter_{round_name}.png** - RMSD scatter plots for each round
2. **quality_distributions_{round_name}.png** - Quality metric distributions
3. **quality_summary_report.png** - 2x2 comprehensive quality summary
4. **metric_correlation_heatmap.png** - Correlation analysis between metrics
5. **pipeline_progress.png** - Pipeline stage progression
6. **msa_evolution_analysis.png** - MSA evolution through stages
7. **analysis_report.html** - Interactive HTML report

## Configuration Options

The `HitExpandPlottingConfig` supports:

```python
@dataclass
class HitExpandPlottingConfig:
    # Plot dimensions
    figsize: Tuple[int, int] = (15, 7)
    dpi: int = 300
    
    # Colors
    initial_color: str = "#87CEEB"
    end_color: str = "#FFFFFF"
    
    # Thresholds
    plddt_threshold: float = 75.0
    filter_criteria_threshold: float = 0.8
    
    # Plot generation flags
    generate_quality_plots: bool = True
    generate_scatter_plots: bool = True
    generate_distribution_plots: bool = True
    generate_summary_plots: bool = True
```

## Benefits Over Basic Plotting

1. **Multi-round Analysis**: Tracks progress across pipeline stages
2. **Statistical Overlays**: Automatic mean, median, and threshold visualization
3. **Professional Presentation**: Publication-ready plots with proper styling
4. **Comprehensive Reporting**: HTML reports for easy sharing
5. **MSA Evolution Tracking**: Visualizes how MSA changes through pipeline
6. **Correlation Analysis**: Identifies relationships between quality metrics

## Dependencies

- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- af_claseq.utils (for structure analysis and plotting utilities)

## Future Enhancements

1. Interactive plots using plotly
2. 3D structure visualization integration
3. Sequence logo generation for conserved regions
4. Network visualization for sequence relationships
5. Real-time monitoring dashboard