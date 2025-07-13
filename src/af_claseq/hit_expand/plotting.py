"""
Plotting and visualization module for hit expand pipeline.

This module provides the HitExpandPlotter class that generates
visualizations and analysis plots for the hit expand pipeline results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from af_claseq.hit_expand.config import HitExpandPlottingConfig
from af_claseq.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HitExpandPlotter:
    """Generates plots and visualizations for hit expand pipeline results."""
    
    def __init__(self, base_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the hit expand plotter.
        
        Args:
            base_dir: Base directory for outputs
            logger: Optional logger instance
        """
        self.base_dir = Path(base_dir)
        self.logger = logger or get_logger(__name__)
        
        # Set up matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger.info(f"HitExpandPlotter initialized for base directory: {self.base_dir}")
    
    def create_hit_expand_plots(
        self, 
        msa_output: Path, 
        config_file: str, 
        plots_dir: Path,
        plot_config: Optional[HitExpandPlottingConfig] = None
    ) -> Dict[str, Path]:
        """
        Create comprehensive plots for hit expand results.
        
        Args:
            msa_output: Path to MSA pipeline output
            config_file: Path to AF-ClaSeq config file
            plots_dir: Directory to save plots
            plot_config: Optional plotting configuration
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        if plot_config is None:
            plot_config = HitExpandPlottingConfig()
        
        plots_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = {}
        
        try:
            self.logger.info("Creating hit expand analysis plots")
            
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Find analysis results
            analysis_results = self._find_analysis_results()
            
            if analysis_results:
                # Create quality distribution plots
                if plot_config.generate_quality_plots:
                    quality_plots = self._create_quality_plots(analysis_results, plots_dir, plot_config)
                    saved_plots.update(quality_plots)
                
                # Create scatter plots
                if plot_config.generate_scatter_plots:
                    scatter_plots = self._create_scatter_plots(analysis_results, plots_dir, plot_config)
                    saved_plots.update(scatter_plots)
                
                # Create distribution plots
                if plot_config.generate_distribution_plots:
                    dist_plots = self._create_distribution_plots(analysis_results, plots_dir, plot_config)
                    saved_plots.update(dist_plots)
                
                # Create summary plots
                if plot_config.generate_summary_plots:
                    summary_plots = self._create_summary_plots(analysis_results, plots_dir, plot_config)
                    saved_plots.update(summary_plots)
            
            # Create MSA comparison plots
            msa_plots = self._create_msa_comparison_plots(msa_output, plots_dir, plot_config)
            saved_plots.update(msa_plots)
            
            # Create pipeline overview plot
            overview_plot = self._create_pipeline_overview_plot(plots_dir, plot_config)
            if overview_plot:
                saved_plots['pipeline_overview'] = overview_plot
            
            self.logger.info(f"Created {len(saved_plots)} hit expand plots")
            return saved_plots
            
        except Exception as e:
            self.logger.error(f"Error creating hit expand plots: {str(e)}", exc_info=True)
            return {}
    
    def _find_analysis_results(self) -> Optional[pd.DataFrame]:
        """Find and load structure analysis results."""
        try:
            # Look for analysis results in multiple locations
            possible_locations = [
                self.base_dir / "01_msa_pipeline" / "structure_analysis_results.csv",
                self.base_dir / "02_analysis" / "structure_analysis_results.csv",
                self.base_dir / "structure_analysis_results.csv"
            ]
            
            for results_file in possible_locations:
                if results_file.exists():
                    self.logger.info(f"Found analysis results: {results_file}")
                    return pd.read_csv(results_file)
            
            self.logger.warning("No structure analysis results found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading analysis results: {str(e)}", exc_info=True)
            return None
    
    def _create_quality_plots(
        self, 
        results_df: pd.DataFrame, 
        plots_dir: Path, 
        config: HitExpandPlottingConfig
    ) -> Dict[str, Path]:
        """Create quality distribution plots."""
        plots = {}
        
        try:
            # pLDDT distribution plot
            if 'plddt' in results_df.columns:
                fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
                
                ax.hist(results_df['plddt'], bins=50, alpha=0.7, color=config.initial_color, edgecolor='black')
                ax.axvline(config.plddt_threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Threshold ({config.plddt_threshold})')
                
                ax.set_xlabel('pLDDT Score')
                ax.set_ylabel('Frequency')
                ax.set_title('pLDDT Score Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_file = plots_dir / 'plddt_distribution.png'
                plt.savefig(plot_file, dpi=config.dpi, bbox_inches='tight')
                plt.close()
                
                plots['plddt_distribution'] = plot_file
                self.logger.info(f"Created pLDDT distribution plot: {plot_file}")
            
            # Additional quality metrics
            quality_columns = [col for col in results_df.columns if col not in ['pdb_file', 'a3m_file']]
            
            if len(quality_columns) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(config.figsize[0]*2, config.figsize[1]*2), dpi=config.dpi)
                axes = axes.flatten()
                
                for i, col in enumerate(quality_columns[:4]):
                    if i < len(axes):
                        axes[i].hist(results_df[col], bins=30, alpha=0.7, color=config.initial_color, edgecolor='black')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                        axes[i].set_title(f'{col} Distribution')
                        axes[i].grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(len(quality_columns), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                plot_file = plots_dir / 'quality_metrics_distribution.png'
                plt.savefig(plot_file, dpi=config.dpi, bbox_inches='tight')
                plt.close()
                
                plots['quality_metrics_distribution'] = plot_file
                self.logger.info(f"Created quality metrics distribution plot: {plot_file}")
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating quality plots: {str(e)}", exc_info=True)
            return {}
    
    def _create_scatter_plots(
        self, 
        results_df: pd.DataFrame, 
        plots_dir: Path, 
        config: HitExpandPlottingConfig
    ) -> Dict[str, Path]:
        """Create scatter plots for metric relationships."""
        plots = {}
        
        try:
            numeric_columns = results_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) >= 2:
                # Create pairwise scatter plots
                fig, axes = plt.subplots(2, 2, figsize=(config.figsize[0]*2, config.figsize[1]*2), dpi=config.dpi)
                axes = axes.flatten()
                
                combinations = [(numeric_columns[0], numeric_columns[1])]
                if len(numeric_columns) > 2:
                    combinations.extend([
                        (numeric_columns[0], numeric_columns[2]),
                        (numeric_columns[1], numeric_columns[2]),
                        (numeric_columns[0], numeric_columns[1])  # Repeat if needed
                    ])
                
                for i, (col1, col2) in enumerate(combinations[:4]):
                    if i < len(axes):
                        scatter = axes[i].scatter(results_df[col1], results_df[col2], 
                                                alpha=0.6, c=results_df.index, cmap='viridis')
                        axes[i].set_xlabel(col1)
                        axes[i].set_ylabel(col2)
                        axes[i].set_title(f'{col1} vs {col2}')
                        axes[i].grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(len(combinations), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                plot_file = plots_dir / 'metric_scatter_plots.png'
                plt.savefig(plot_file, dpi=config.dpi, bbox_inches='tight')
                plt.close()
                
                plots['metric_scatter_plots'] = plot_file
                self.logger.info(f"Created scatter plots: {plot_file}")
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plots: {str(e)}", exc_info=True)
            return {}
    
    def _create_distribution_plots(
        self, 
        results_df: pd.DataFrame, 
        plots_dir: Path, 
        config: HitExpandPlottingConfig
    ) -> Dict[str, Path]:
        """Create distribution comparison plots."""
        plots = {}
        
        try:
            # Create box plots for all numeric columns
            numeric_columns = results_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_columns:
                fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
                
                # Normalize data for better visualization
                normalized_data = []
                labels = []
                
                for col in numeric_columns:
                    data = results_df[col].dropna()
                    if len(data) > 0:
                        # Normalize to 0-1 range
                        normalized = (data - data.min()) / (data.max() - data.min()) if data.max() > data.min() else data
                        normalized_data.append(normalized)
                        labels.append(col)
                
                if normalized_data:
                    bp = ax.boxplot(normalized_data, labels=labels, patch_artist=True)
                    
                    # Color the boxes
                    for patch in bp['boxes']:
                        patch.set_facecolor(config.initial_color)
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel('Normalized Values')
                    ax.set_title('Distribution of Quality Metrics (Normalized)')
                    ax.grid(True, alpha=0.3)
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    plot_file = plots_dir / 'metric_distributions.png'
                    plt.savefig(plot_file, dpi=config.dpi, bbox_inches='tight')
                    plt.close()
                    
                    plots['metric_distributions'] = plot_file
                    self.logger.info(f"Created distribution plots: {plot_file}")
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plots: {str(e)}", exc_info=True)
            return {}
    
    def _create_summary_plots(
        self, 
        results_df: pd.DataFrame, 
        plots_dir: Path, 
        config: HitExpandPlottingConfig
    ) -> Dict[str, Path]:
        """Create summary overview plots."""
        plots = {}
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(config.figsize[0]*2, config.figsize[1]*2), dpi=config.dpi)
            
            # 1. Structure count and quality summary
            total_structures = len(results_df)
            if 'plddt' in results_df.columns:
                high_quality = (results_df['plddt'] >= config.plddt_threshold).sum()
                low_quality = total_structures - high_quality
                
                ax1.pie([high_quality, low_quality], labels=['High Quality', 'Low Quality'], 
                       autopct='%1.1f%%', colors=[config.initial_color, config.end_color])
                ax1.set_title(f'Structure Quality Distribution\n(Threshold: {config.plddt_threshold})')
            
            # 2. Metric correlation heatmap
            numeric_columns = results_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) > 1:
                correlation_matrix = results_df[numeric_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2)
                ax2.set_title('Metric Correlation Heatmap')
            
            # 3. Top structures by pLDDT
            if 'plddt' in results_df.columns:
                top_structures = results_df.nlargest(10, 'plddt')
                ax3.barh(range(len(top_structures)), top_structures['plddt'], color=config.initial_color)
                ax3.set_xlabel('pLDDT Score')
                ax3.set_ylabel('Structure Rank')
                ax3.set_title('Top 10 Structures by pLDDT')
                ax3.grid(True, alpha=0.3)
            
            # 4. Summary statistics
            ax4.axis('off')
            
            # Create summary text
            summary_text = f"Hit Expand Pipeline Summary\n\n"
            summary_text += f"Total Structures: {total_structures}\n"
            
            if 'plddt' in results_df.columns:
                summary_text += f"Mean pLDDT: {results_df['plddt'].mean():.2f}\n"
                summary_text += f"High Quality Structures: {high_quality}\n"
                summary_text += f"Quality Rate: {(high_quality/total_structures)*100:.1f}%\n"
            
            # Add other metrics
            for col in numeric_columns:
                if col != 'plddt':
                    summary_text += f"Mean {col}: {results_df[col].mean():.2f}\n"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor=config.initial_color, alpha=0.3))
            
            plt.tight_layout()
            
            plot_file = plots_dir / 'summary_overview.png'
            plt.savefig(plot_file, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            plots['summary_overview'] = plot_file
            self.logger.info(f"Created summary overview plot: {plot_file}")
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating summary plots: {str(e)}", exc_info=True)
            return {}
    
    def _create_msa_comparison_plots(
        self, 
        msa_output: Path, 
        plots_dir: Path, 
        config: HitExpandPlottingConfig
    ) -> Dict[str, Path]:
        """Create MSA comparison plots."""
        plots = {}
        
        try:
            # Basic MSA statistics
            msa_stats = self._analyze_msa_file(msa_output)
            
            if msa_stats:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figsize, dpi=config.dpi)
                
                # Sequence length distribution
                if 'sequence_lengths' in msa_stats:
                    ax1.hist(msa_stats['sequence_lengths'], bins=30, alpha=0.7, 
                            color=config.initial_color, edgecolor='black')
                    ax1.set_xlabel('Sequence Length')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('Sequence Length Distribution')
                    ax1.grid(True, alpha=0.3)
                
                # MSA composition
                if 'composition' in msa_stats:
                    amino_acids = list(msa_stats['composition'].keys())
                    frequencies = list(msa_stats['composition'].values())
                    
                    ax2.bar(amino_acids, frequencies, color=config.initial_color, alpha=0.7)
                    ax2.set_xlabel('Amino Acid')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('MSA Amino Acid Composition')
                    ax2.grid(True, alpha=0.3)
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                
                plot_file = plots_dir / 'msa_analysis.png'
                plt.savefig(plot_file, dpi=config.dpi, bbox_inches='tight')
                plt.close()
                
                plots['msa_analysis'] = plot_file
                self.logger.info(f"Created MSA analysis plot: {plot_file}")
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating MSA comparison plots: {str(e)}", exc_info=True)
            return {}
    
    def _create_pipeline_overview_plot(self, plots_dir: Path, config: HitExpandPlottingConfig) -> Optional[Path]:
        """Create a pipeline overview flowchart."""
        try:
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create a simple pipeline flowchart
            steps = [
                "Input MSA",
                "MMseqs2 Clustering",
                "Subset Generation",
                "Structure Prediction",
                "Structure Analysis",
                "Hit Expansion",
                "Final MSA"
            ]
            
            y_positions = np.arange(len(steps))
            
            # Create boxes for each step
            for i, step in enumerate(steps):
                rect = plt.Rectangle((0, i-0.3), 2, 0.6, facecolor=config.initial_color, 
                                   edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
                ax.text(1, i, step, ha='center', va='center', fontsize=10, fontweight='bold')
                
                # Add arrows between steps
                if i < len(steps) - 1:
                    ax.arrow(1, i+0.3, 0, 0.4, head_width=0.1, head_length=0.1, 
                            fc='black', ec='black')
            
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-0.5, len(steps)-0.5)
            ax.set_title('Hit Expand Pipeline Overview', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            plot_file = plots_dir / 'pipeline_overview.png'
            plt.savefig(plot_file, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Created pipeline overview plot: {plot_file}")
            return plot_file
            
        except Exception as e:
            self.logger.error(f"Error creating pipeline overview plot: {str(e)}", exc_info=True)
            return None
    
    def _analyze_msa_file(self, msa_file: Path) -> Dict[str, Any]:
        """Analyze MSA file and return statistics."""
        try:
            sequences = []
            headers = []
            
            with open(msa_file, 'r') as f:
                current_seq = ""
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append(current_seq)
                            current_seq = ""
                        headers.append(line)
                    else:
                        current_seq += line
                
                if current_seq:
                    sequences.append(current_seq)
            
            # Calculate statistics
            sequence_lengths = [len(seq) for seq in sequences]
            
            # Amino acid composition
            all_sequences = ''.join(sequences)
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            composition = {}
            
            for aa in amino_acids:
                composition[aa] = all_sequences.count(aa)
            
            return {
                'num_sequences': len(sequences),
                'sequence_lengths': sequence_lengths,
                'mean_length': np.mean(sequence_lengths) if sequence_lengths else 0,
                'composition': composition
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing MSA file: {str(e)}", exc_info=True)
            return {}