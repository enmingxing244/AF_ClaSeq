"""
Enhanced plotting and visualization module for hit expand pipeline.

This module incorporates advanced plotting capabilities from the MSA pipeline,
including multi-metric analysis, threshold visualization, and comprehensive
quality summaries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from af_claseq.hit_expand.config import HitExpandPlottingConfig
from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.structure_analysis import StructureAnalyzer

# Try to import af_claseq plotting utilities
try:
    from af_claseq.utils.plotting_manager import PlottingManager, COLORS
    USE_AF_CLASEQ_PLOTTING = True
except ImportError:
    USE_AF_CLASEQ_PLOTTING = False
    COLORS = {
        'primary': '#87CEEB',
        'secondary': '#FF6B6B',
        'success': '#4CAF50',
        'warning': '#FFA726',
        'info': '#42A5F5'
    }

logger = get_logger(__name__)


class EnhancedHitExpandPlotter:
    """Enhanced plotter with MSA pipeline-inspired visualizations."""
    
    def __init__(self, base_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the enhanced hit expand plotter.
        
        Args:
            base_dir: Base directory for outputs
            logger: Optional logger instance
        """
        self.base_dir = Path(base_dir)
        self.logger = logger or get_logger(__name__)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = sns.color_palette("husl", 8)
        
        # Initialize structure analyzer
        self.structure_analyzer = StructureAnalyzer()
        
        # Initialize plotting manager if available
        if USE_AF_CLASEQ_PLOTTING:
            self.plotting_manager = PlottingManager()
        else:
            self.plotting_manager = None
        
        self.logger.info(f"EnhancedHitExpandPlotter initialized for base directory: {self.base_dir}")
    
    def create_comprehensive_analysis_plots(
        self, 
        msa_output: Path, 
        config_file: str, 
        plots_dir: Path,
        plot_config: Optional[HitExpandPlottingConfig] = None
    ) -> Dict[str, Path]:
        """
        Create comprehensive plots inspired by MSA pipeline.
        
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
            self.logger.info("Creating comprehensive hit expand analysis plots")
            
            # Load configuration
            with open(config_file, 'r') as f:
                af_claseq_config = json.load(f)
            
            # Find and load analysis results
            analysis_results = self._load_all_analysis_results()
            
            if analysis_results:
                # 1. Create RMSD scatter plots with pLDDT coloring (MSA pipeline style)
                scatter_plots = self._create_rmsd_scatter_plots(
                    analysis_results, af_claseq_config, plots_dir, plot_config
                )
                saved_plots.update(scatter_plots)
                
                # 2. Create quality distribution plots with KDE
                quality_plots = self._create_quality_distribution_plots(
                    analysis_results, plots_dir, plot_config
                )
                saved_plots.update(quality_plots)
                
                # 3. Create 2x2 quality summary plot (MSA pipeline signature)
                summary_plot = self._create_quality_summary_plot(
                    analysis_results, plots_dir, plot_config
                )
                if summary_plot:
                    saved_plots['quality_summary'] = summary_plot
                
                # 4. Create metric correlation heatmap
                correlation_plot = self._create_metric_correlation_plot(
                    analysis_results, plots_dir, plot_config
                )
                if correlation_plot:
                    saved_plots['metric_correlation'] = correlation_plot
                
                # 5. Create pipeline progress visualization
                progress_plot = self._create_pipeline_progress_plot(
                    analysis_results, plots_dir, plot_config
                )
                if progress_plot:
                    saved_plots['pipeline_progress'] = progress_plot
            
            # 6. Create MSA evolution plots
            evolution_plots = self._create_msa_evolution_plots(
                msa_output, plots_dir, plot_config
            )
            saved_plots.update(evolution_plots)
            
            # 7. Generate comprehensive HTML report
            report_path = self._generate_html_report(saved_plots, analysis_results, plots_dir)
            if report_path:
                saved_plots['html_report'] = report_path
            
            self.logger.info(f"Created {len(saved_plots)} comprehensive analysis plots")
            return saved_plots
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive plots: {str(e)}", exc_info=True)
            return saved_plots
    
    def _load_all_analysis_results(self) -> Dict[str, pd.DataFrame]:
        """Load analysis results from all rounds."""
        results = {}
        
        # Check for multiple rounds of analysis
        rounds = ['00_initial_prediction', '01_hit_expansion', '02_final_optimization']
        
        for round_name in rounds:
            round_dir = self.base_dir / round_name
            if round_dir.exists():
                results_file = round_dir / 'structure_analysis_results.csv'
                if results_file.exists():
                    results[round_name] = pd.read_csv(results_file)
                    self.logger.info(f"Loaded {len(results[round_name])} results from {round_name}")
        
        # Also check for single-round results
        main_results = self.base_dir / 'structure_analysis_results.csv'
        if main_results.exists() and not results:
            results['main'] = pd.read_csv(main_results)
        
        return results
    
    def _create_rmsd_scatter_plots(
        self,
        results: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        plots_dir: Path,
        plot_config: HitExpandPlottingConfig
    ) -> Dict[str, Path]:
        """Create RMSD scatter plots with pLDDT coloring (MSA pipeline style)."""
        saved_plots = {}
        
        try:
            # Get filter criteria from config
            filter_criteria = config.get('filter_criteria', {})
            
            for round_name, df in results.items():
                if df.empty:
                    continue
                
                # Find RMSD columns
                rmsd_cols = [col for col in df.columns if 'rmsd' in col.lower()]
                
                if rmsd_cols and 'plddt' in df.columns:
                    fig, axes = plt.subplots(1, len(rmsd_cols), 
                                           figsize=(6*len(rmsd_cols), 6))
                    if len(rmsd_cols) == 1:
                        axes = [axes]
                    
                    for idx, rmsd_col in enumerate(rmsd_cols):
                        ax = axes[idx]
                        
                        # Create scatter plot
                        scatter = ax.scatter(
                            df.index,
                            df[rmsd_col],
                            c=df['plddt'],
                            cmap='viridis',
                            alpha=0.6,
                            edgecolors='black',
                            linewidth=0.5
                        )
                        
                        # Add threshold line if available
                        criterion_name = rmsd_col.replace('_rmsd', '')
                        if criterion_name in filter_criteria:
                            threshold = filter_criteria[criterion_name].get('threshold', None)
                            if threshold:
                                ax.axhline(y=threshold, color='red', linestyle='--', 
                                         linewidth=2, label=f'Threshold: {threshold}')
                        
                        # Add mean and median lines
                        mean_val = df[rmsd_col].mean()
                        median_val = df[rmsd_col].median()
                        ax.axhline(y=mean_val, color='orange', linestyle='-', 
                                 alpha=0.7, label=f'Mean: {mean_val:.2f}')
                        ax.axhline(y=median_val, color='green', linestyle='-', 
                                 alpha=0.7, label=f'Median: {median_val:.2f}')
                        
                        ax.set_xlabel('Structure Index')
                        ax.set_ylabel(f'{rmsd_col} (Å)')
                        ax.set_title(f'{rmsd_col} vs Structure Index\n{round_name}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Add colorbar
                        cbar = plt.colorbar(scatter, ax=ax)
                        cbar.set_label('pLDDT Score')
                    
                    plt.tight_layout()
                    
                    plot_file = plots_dir / f'rmsd_scatter_{round_name}.png'
                    plt.savefig(plot_file, dpi=plot_config.dpi, bbox_inches='tight')
                    plt.close()
                    
                    saved_plots[f'rmsd_scatter_{round_name}'] = plot_file
                    self.logger.info(f"Created RMSD scatter plot for {round_name}")
            
            return saved_plots
            
        except Exception as e:
            self.logger.error(f"Error creating RMSD scatter plots: {str(e)}", exc_info=True)
            return saved_plots
    
    def _create_quality_distribution_plots(
        self,
        results: Dict[str, pd.DataFrame],
        plots_dir: Path,
        plot_config: HitExpandPlottingConfig
    ) -> Dict[str, Path]:
        """Create quality distribution plots with KDE overlays."""
        saved_plots = {}
        
        try:
            for round_name, df in results.items():
                if df.empty or 'plddt' not in df.columns:
                    continue
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                # 1. pLDDT distribution with KDE
                ax = axes[0]
                ax.hist(df['plddt'], bins=30, alpha=0.7, density=True, 
                       color=COLORS.get('primary', 'skyblue'), edgecolor='black')
                df['plddt'].plot.kde(ax=ax, color='red', linewidth=2)
                ax.axvline(plot_config.plddt_threshold, color='red', linestyle='--', 
                         linewidth=2, label=f'Threshold: {plot_config.plddt_threshold}')
                ax.set_xlabel('pLDDT Score')
                ax.set_ylabel('Density')
                ax.set_title('pLDDT Distribution with KDE')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 2. RMSD distributions
                rmsd_cols = [col for col in df.columns if 'rmsd' in col.lower()]
                if rmsd_cols:
                    ax = axes[1]
                    for rmsd_col in rmsd_cols[:3]:  # Limit to 3 for clarity
                        df[rmsd_col].plot.kde(ax=ax, label=rmsd_col, linewidth=2)
                    ax.set_xlabel('RMSD (Å)')
                    ax.set_ylabel('Density')
                    ax.set_title('RMSD Distributions')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # 3. Quality categories pie chart
                ax = axes[2]
                high_quality = (df['plddt'] >= plot_config.plddt_threshold).sum()
                low_quality = len(df) - high_quality
                ax.pie([high_quality, low_quality], 
                      labels=['High Quality', 'Low Quality'],
                      autopct='%1.1f%%',
                      colors=[COLORS.get('success', 'green'), COLORS.get('warning', 'orange')],
                      startangle=90)
                ax.set_title(f'Quality Distribution\n(pLDDT ≥ {plot_config.plddt_threshold})')
                
                # 4. Box plots for all metrics
                ax = axes[3]
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    # Normalize for visualization
                    normalized_data = []
                    labels = []
                    for col in numeric_cols[:5]:  # Limit to 5 metrics
                        data = df[col].dropna()
                        if len(data) > 0 and data.max() > data.min():
                            normalized = (data - data.min()) / (data.max() - data.min())
                            normalized_data.append(normalized)
                            labels.append(col)
                    
                    if normalized_data:
                        bp = ax.boxplot(normalized_data, labels=labels, patch_artist=True)
                        for patch in bp['boxes']:
                            patch.set_facecolor(COLORS.get('info', 'lightblue'))
                            patch.set_alpha(0.7)
                        ax.set_ylabel('Normalized Value')
                        ax.set_title('Metric Distributions (Normalized)')
                        ax.grid(True, alpha=0.3)
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                
                plot_file = plots_dir / f'quality_distributions_{round_name}.png'
                plt.savefig(plot_file, dpi=plot_config.dpi, bbox_inches='tight')
                plt.close()
                
                saved_plots[f'quality_distributions_{round_name}'] = plot_file
                self.logger.info(f"Created quality distribution plots for {round_name}")
            
            return saved_plots
            
        except Exception as e:
            self.logger.error(f"Error creating quality distribution plots: {str(e)}", exc_info=True)
            return saved_plots
    
    def _create_quality_summary_plot(
        self,
        results: Dict[str, pd.DataFrame],
        plots_dir: Path,
        plot_config: HitExpandPlottingConfig
    ) -> Optional[Path]:
        """Create 2x2 quality summary plot (MSA pipeline signature)."""
        try:
            # Use the most recent or complete results
            df = None
            for round_name in ['02_final_optimization', '01_hit_expansion', '00_initial_prediction', 'main']:
                if round_name in results and not results[round_name].empty:
                    df = results[round_name]
                    break
            
            if df is None or df.empty:
                return None
            
            fig = plt.figure(figsize=(14, 12))
            gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)
            
            # 1. pLDDT vs primary RMSD scatter
            ax1 = fig.add_subplot(gs[0, 0])
            rmsd_cols = [col for col in df.columns if 'rmsd' in col.lower()]
            if rmsd_cols and 'plddt' in df.columns:
                primary_rmsd = rmsd_cols[0]
                scatter = ax1.scatter(df[primary_rmsd], df['plddt'], 
                                    c=df.index, cmap='viridis', alpha=0.6)
                ax1.axhline(plot_config.plddt_threshold, color='red', linestyle='--', 
                          linewidth=2, label=f'pLDDT threshold: {plot_config.plddt_threshold}')
                ax1.set_xlabel(f'{primary_rmsd} (Å)')
                ax1.set_ylabel('pLDDT Score')
                ax1.set_title('pLDDT vs RMSD')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Quality category bar chart
            ax2 = fig.add_subplot(gs[0, 1])
            if 'plddt' in df.columns:
                quality_bins = [0, 50, 70, 90, 100]
                quality_labels = ['Very Low\n(0-50)', 'Low\n(50-70)', 
                                'Confident\n(70-90)', 'Very High\n(90-100)']
                quality_counts = pd.cut(df['plddt'], bins=quality_bins, 
                                      labels=quality_labels).value_counts()
                
                bars = ax2.bar(quality_counts.index, quality_counts.values, 
                              color=['red', 'orange', 'yellow', 'green'], 
                              edgecolor='black', alpha=0.7)
                ax2.set_xlabel('pLDDT Category')
                ax2.set_ylabel('Count')
                ax2.set_title('Structure Quality Categories')
                
                # Add count labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
            
            # 3. pLDDT distribution histogram
            ax3 = fig.add_subplot(gs[1, 0])
            if 'plddt' in df.columns:
                n, bins, patches = ax3.hist(df['plddt'], bins=30, alpha=0.7, 
                                          color=COLORS.get('primary', 'skyblue'), 
                                          edgecolor='black')
                
                # Color bins by quality
                for i, patch in enumerate(patches):
                    if bins[i] < 50:
                        patch.set_facecolor('red')
                    elif bins[i] < 70:
                        patch.set_facecolor('orange')
                    elif bins[i] < 90:
                        patch.set_facecolor('yellow')
                    else:
                        patch.set_facecolor('green')
                
                ax3.axvline(df['plddt'].mean(), color='blue', linestyle='-', 
                          linewidth=2, label=f'Mean: {df["plddt"].mean():.1f}')
                ax3.axvline(df['plddt'].median(), color='green', linestyle='-', 
                          linewidth=2, label=f'Median: {df["plddt"].median():.1f}')
                ax3.set_xlabel('pLDDT Score')
                ax3.set_ylabel('Frequency')
                ax3.set_title('pLDDT Score Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. Summary statistics table
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Create summary statistics
            stats_data = []
            stats_data.append(['Total Structures', f'{len(df):,}'])
            
            if 'plddt' in df.columns:
                stats_data.append(['Mean pLDDT', f'{df["plddt"].mean():.2f}'])
                stats_data.append(['Median pLDDT', f'{df["plddt"].median():.2f}'])
                stats_data.append(['High Quality (≥70)', 
                                 f'{(df["plddt"] >= 70).sum():,} ({(df["plddt"] >= 70).sum()/len(df)*100:.1f}%)'])
            
            # Add RMSD statistics
            if rmsd_cols:
                for rmsd_col in rmsd_cols[:2]:  # Show top 2 RMSD metrics
                    stats_data.append([f'Mean {rmsd_col}', f'{df[rmsd_col].mean():.2f} Å'])
            
            # Create table
            table = ax4.table(cellText=stats_data, 
                            colLabels=['Metric', 'Value'],
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(stats_data) + 1):
                for j in range(2):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
            
            # Overall title
            fig.suptitle('Hit Expand Quality Summary Report', fontsize=16, fontweight='bold')
            
            plot_file = plots_dir / 'quality_summary_report.png'
            plt.savefig(plot_file, dpi=plot_config.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Created quality summary report: {plot_file}")
            return plot_file
            
        except Exception as e:
            self.logger.error(f"Error creating quality summary plot: {str(e)}", exc_info=True)
            return None
    
    def _create_metric_correlation_plot(
        self,
        results: Dict[str, pd.DataFrame],
        plots_dir: Path,
        plot_config: HitExpandPlottingConfig
    ) -> Optional[Path]:
        """Create metric correlation heatmap."""
        try:
            # Combine all results for correlation analysis
            all_results = []
            for round_name, df in results.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy['round'] = round_name
                    all_results.append(df_copy)
            
            if not all_results:
                return None
            
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Select numeric columns
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['index', 'round']]
            
            if len(numeric_cols) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = combined_df[numeric_cols].corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, 
                       linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
            
            ax.set_title('Metric Correlation Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            plot_file = plots_dir / 'metric_correlation_heatmap.png'
            plt.savefig(plot_file, dpi=plot_config.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Created metric correlation heatmap: {plot_file}")
            return plot_file
            
        except Exception as e:
            self.logger.error(f"Error creating metric correlation plot: {str(e)}", exc_info=True)
            return None
    
    def _create_pipeline_progress_plot(
        self,
        results: Dict[str, pd.DataFrame],
        plots_dir: Path,
        plot_config: HitExpandPlottingConfig
    ) -> Optional[Path]:
        """Create pipeline progress visualization."""
        try:
            if not results:
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 1. Structure count by round
            rounds = []
            counts = []
            high_quality_counts = []
            
            round_order = ['00_initial_prediction', '01_hit_expansion', '02_final_optimization']
            for round_name in round_order:
                if round_name in results:
                    df = results[round_name]
                    rounds.append(round_name.replace('_', '\n'))
                    counts.append(len(df))
                    if 'plddt' in df.columns:
                        high_quality_counts.append((df['plddt'] >= plot_config.plddt_threshold).sum())
                    else:
                        high_quality_counts.append(0)
            
            if rounds:
                x = np.arange(len(rounds))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, counts, width, label='Total Structures', 
                               color=COLORS.get('primary', 'skyblue'), alpha=0.7)
                bars2 = ax1.bar(x + width/2, high_quality_counts, width, 
                               label='High Quality', color=COLORS.get('success', 'green'), alpha=0.7)
                
                ax1.set_xlabel('Pipeline Stage')
                ax1.set_ylabel('Number of Structures')
                ax1.set_title('Structure Count Progress Through Pipeline')
                ax1.set_xticks(x)
                ax1.set_xticklabels(rounds)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
            
            # 2. Quality metrics evolution
            if len(results) > 1:
                metric_evolution = {}
                
                for round_name in round_order:
                    if round_name in results:
                        df = results[round_name]
                        if 'plddt' in df.columns:
                            if 'plddt' not in metric_evolution:
                                metric_evolution['plddt'] = []
                            metric_evolution['plddt'].append(df['plddt'].mean())
                        
                        # Add RMSD metrics
                        rmsd_cols = [col for col in df.columns if 'rmsd' in col.lower()]
                        for rmsd_col in rmsd_cols[:2]:  # Limit to 2 RMSD metrics
                            if rmsd_col not in metric_evolution:
                                metric_evolution[rmsd_col] = []
                            metric_evolution[rmsd_col].append(df[rmsd_col].mean())
                
                # Plot evolution
                for metric, values in metric_evolution.items():
                    if len(values) > 1:
                        ax2.plot(range(len(values)), values, marker='o', 
                               label=metric, linewidth=2, markersize=8)
                
                ax2.set_xlabel('Pipeline Stage')
                ax2.set_ylabel('Metric Value')
                ax2.set_title('Quality Metrics Evolution')
                ax2.set_xticks(range(len(rounds)))
                ax2.set_xticklabels(rounds)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_file = plots_dir / 'pipeline_progress.png'
            plt.savefig(plot_file, dpi=plot_config.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Created pipeline progress plot: {plot_file}")
            return plot_file
            
        except Exception as e:
            self.logger.error(f"Error creating pipeline progress plot: {str(e)}", exc_info=True)
            return None
    
    def _create_msa_evolution_plots(
        self,
        msa_output: Path,
        plots_dir: Path,
        plot_config: HitExpandPlottingConfig
    ) -> Dict[str, Path]:
        """Create MSA evolution and comparison plots."""
        saved_plots = {}
        
        try:
            # Find all MSA files in the pipeline
            msa_files = []
            msa_labels = []
            
            # Check for MSAs at different stages
            stage_paths = [
                (self.base_dir / "00_initial_msa.a3m", "Initial MSA"),
                (self.base_dir / "01_clustered_msa.a3m", "After Clustering"),
                (self.base_dir / "02_expanded_msa.a3m", "After Hit Expansion"),
                (msa_output, "Final MSA")
            ]
            
            for msa_path, label in stage_paths:
                if msa_path.exists():
                    msa_files.append(msa_path)
                    msa_labels.append(label)
            
            if len(msa_files) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                axes = axes.flatten()
                
                # Analyze each MSA
                msa_stats = []
                for msa_file in msa_files:
                    stats = self._analyze_msa_file(msa_file)
                    msa_stats.append(stats)
                
                # 1. Sequence count evolution
                ax = axes[0]
                seq_counts = [stats['num_sequences'] for stats in msa_stats]
                bars = ax.bar(msa_labels, seq_counts, color=COLORS.get('primary', 'skyblue'), 
                             edgecolor='black', alpha=0.7)
                ax.set_xlabel('MSA Stage')
                ax.set_ylabel('Number of Sequences')
                ax.set_title('Sequence Count Evolution')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, count in zip(bars, seq_counts):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{count:,}', ha='center', va='bottom')
                
                # 2. Sequence length distribution comparison
                ax = axes[1]
                for stats, label in zip(msa_stats, msa_labels):
                    if 'sequence_lengths' in stats:
                        ax.hist(stats['sequence_lengths'], bins=30, alpha=0.5, 
                               label=label, density=True)
                ax.set_xlabel('Sequence Length')
                ax.set_ylabel('Density')
                ax.set_title('Sequence Length Distribution Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 3. Sequence diversity (unique sequences)
                ax = axes[2]
                diversity_ratios = []
                for stats in msa_stats:
                    if 'unique_sequences' in stats and stats['num_sequences'] > 0:
                        ratio = stats['unique_sequences'] / stats['num_sequences']
                        diversity_ratios.append(ratio * 100)
                    else:
                        diversity_ratios.append(0)
                
                if diversity_ratios:
                    bars = ax.bar(msa_labels, diversity_ratios, 
                                 color=COLORS.get('info', 'lightblue'), 
                                 edgecolor='black', alpha=0.7)
                    ax.set_xlabel('MSA Stage')
                    ax.set_ylabel('Unique Sequences (%)')
                    ax.set_title('Sequence Diversity')
                    ax.grid(True, alpha=0.3)
                
                # 4. Gap content analysis
                ax = axes[3]
                gap_percentages = []
                for stats in msa_stats:
                    if 'gap_percentage' in stats:
                        gap_percentages.append(stats['gap_percentage'])
                    else:
                        gap_percentages.append(0)
                
                if gap_percentages:
                    ax.plot(msa_labels, gap_percentages, marker='o', 
                           color=COLORS.get('warning', 'orange'), 
                           linewidth=2, markersize=10)
                    ax.set_xlabel('MSA Stage')
                    ax.set_ylabel('Gap Percentage (%)')
                    ax.set_title('MSA Gap Content Evolution')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                plot_file = plots_dir / 'msa_evolution_analysis.png'
                plt.savefig(plot_file, dpi=plot_config.dpi, bbox_inches='tight')
                plt.close()
                
                saved_plots['msa_evolution'] = plot_file
                self.logger.info(f"Created MSA evolution analysis: {plot_file}")
            
            return saved_plots
            
        except Exception as e:
            self.logger.error(f"Error creating MSA evolution plots: {str(e)}", exc_info=True)
            return saved_plots
    
    def _analyze_msa_file(self, msa_file: Path) -> Dict[str, Any]:
        """Enhanced MSA file analysis."""
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
            
            # Calculate enhanced statistics
            sequence_lengths = [len(seq.replace('-', '')) for seq in sequences]
            unique_sequences = len(set(sequences))
            
            # Gap analysis
            total_positions = sum(len(seq) for seq in sequences)
            gap_count = sum(seq.count('-') for seq in sequences)
            gap_percentage = (gap_count / total_positions * 100) if total_positions > 0 else 0
            
            # Amino acid composition
            all_sequences = ''.join(sequences).replace('-', '')
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            composition = {aa: all_sequences.count(aa) for aa in amino_acids}
            
            # Conservation analysis (simplified)
            if sequences and len(sequences[0]) > 0:
                conservation_scores = []
                for pos in range(len(sequences[0])):
                    column = [seq[pos] if pos < len(seq) else '-' for seq in sequences]
                    unique_in_column = len(set(column))
                    conservation = 1.0 - (unique_in_column - 1) / 20.0  # Simple conservation metric
                    conservation_scores.append(max(0, conservation))
                mean_conservation = np.mean(conservation_scores)
            else:
                mean_conservation = 0
            
            return {
                'num_sequences': len(sequences),
                'unique_sequences': unique_sequences,
                'sequence_lengths': sequence_lengths,
                'mean_length': np.mean(sequence_lengths) if sequence_lengths else 0,
                'std_length': np.std(sequence_lengths) if sequence_lengths else 0,
                'gap_percentage': gap_percentage,
                'mean_conservation': mean_conservation,
                'composition': composition
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing MSA file: {str(e)}", exc_info=True)
            return {}
    
    def _generate_html_report(
        self,
        plots: Dict[str, Path],
        results: Dict[str, pd.DataFrame],
        plots_dir: Path
    ) -> Optional[Path]:
        """Generate comprehensive HTML report."""
        try:
            html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Hit Expand Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .summary-table th, .summary-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .summary-table th {
            background-color: #4CAF50;
            color: white;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .timestamp {
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hit Expand Pipeline Analysis Report</h1>
        <p class="timestamp">Generated on: {timestamp}</p>
        
        <h2>Pipeline Summary</h2>
        <table class="summary-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
            
            # Add summary statistics
            total_structures = 0
            high_quality_structures = 0
            
            for round_name, df in results.items():
                if not df.empty:
                    total_structures += len(df)
                    if 'plddt' in df.columns:
                        high_quality_structures += (df['plddt'] >= 70).sum()
            
            html_content += f"""
            <tr>
                <td>Total Structures Analyzed</td>
                <td>{total_structures:,}</td>
            </tr>
            <tr>
                <td>High Quality Structures (pLDDT ≥ 70)</td>
                <td>{high_quality_structures:,} ({high_quality_structures/total_structures*100:.1f}%)</td>
            </tr>
            <tr>
                <td>Pipeline Rounds</td>
                <td>{len(results)}</td>
            </tr>
        </table>
        
        <h2>Analysis Plots</h2>
"""
            
            # Add plots
            plot_sections = {
                'quality_summary': 'Quality Summary Report',
                'pipeline_progress': 'Pipeline Progress',
                'metric_correlation': 'Metric Correlations',
                'msa_evolution': 'MSA Evolution',
                'rmsd_scatter': 'RMSD Analysis',
                'quality_distributions': 'Quality Distributions'
            }
            
            for plot_key, title in plot_sections.items():
                matching_plots = [p for k, p in plots.items() if plot_key in k]
                if matching_plots:
                    html_content += f'<h3>{title}</h3>\n'
                    for plot_path in matching_plots:
                        if plot_path.exists():
                            # Use relative path
                            rel_path = plot_path.relative_to(plots_dir.parent)
                            html_content += f"""
        <div class="plot-container">
            <img src="{rel_path}" alt="{title}">
        </div>
"""
            
            html_content += """
    </div>
</body>
</html>
"""
            
            # Save HTML report
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html_content = html_content.format(timestamp=timestamp)
            
            report_path = plots_dir / 'analysis_report.html'
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Generated HTML report: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}", exc_info=True)
            return None