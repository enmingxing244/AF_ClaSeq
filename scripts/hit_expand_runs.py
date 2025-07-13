#!/usr/bin/env python3
"""
Main pipeline runner script for MSA structure optimization.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from af_claseq.hit_expand.config.settings import ConfigManager, PipelineConfig
from af_claseq.hit_expand.core.orchestrator import MSAOptimizationPipeline
from af_claseq.hit_expand.utils.logging_config import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="MSA Structure-Guided Optimization Pipeline (clustering external)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Use existing config file
  python scripts/run_pipeline.py configs/custom.yaml

  # Mode 2: Create config from CLI arguments
  python scripts/run_pipeline.py --input-file /path/to/clustered_reps.a3m --source-msa /path/to/original.a3m --output /path/to/results

  # Generate default configuration template
  python scripts/run_pipeline.py --create-config configs/default.yaml
        """
    )
    
    # Main arguments - Config file mode
    parser.add_argument('config_file', nargs='?', help='Configuration file (YAML/JSON)')
    
    # CLI mode arguments - for creating config from command line
    parser.add_argument('--input-file', help='Input clustered representative A3M file (absolute path)')
    parser.add_argument('--source-msa', help='Source MSA file for similarity search (absolute path)')
    parser.add_argument('-o', '--output', help='Output directory (absolute path)')
    parser.add_argument('--config-file', help='af_claseq configuration file (absolute path)')
    
    # Configuration utilities
    parser.add_argument('--create-config', metavar='FILE',
                       help='Create default configuration file and exit')
    parser.add_argument('--save-config', metavar='FILE',
                       help='Save generated config to file (used with CLI mode)')
    
    # Pipeline control (clustering removed)
    parser.add_argument('--skip-structure-prediction', action='store_true',
                       help='Skip structure prediction step')
    parser.add_argument('--skip-structure-analysis', action='store_true',
                       help='Skip structure analysis step')
    parser.add_argument('--skip-hit-expansion', action='store_true',
                       help='Skip hit expansion step')
    
    # Quick options (clustering options removed)
    parser.add_argument('--num-subsets', type=int, default=2000,
                       help='Number of subsets to generate (default: 2000)')
    parser.add_argument('--subset-size', type=int, default=8,
                       help='Number of sequences per subset (default: 8)')
    parser.add_argument('--num-batches', type=int, default=50,
                       help='Number of batches for processing (default: 50)')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                       help='BLOSUM62 similarity threshold (default: 0.7)')
    
    # Logging and execution
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without executing')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Automatically answer yes to prompts')
    
    return parser


def create_config_from_cli(args: argparse.Namespace) -> PipelineConfig:
    """Create configuration from CLI arguments."""
    config_manager = ConfigManager()
    
    # Create base configuration
    config = config_manager.create_default_config()
    
    # Set required paths from CLI
    config.input_file = args.input_file
    config.source_msa_file = args.source_msa
    config.output_dir = args.output
    
    # Set af_claseq config file if provided
    if args.config_file:
        config.structure_analysis.config_file = args.config_file
    
    # Apply CLI overrides for quick options
    if hasattr(args, 'num_subsets') and args.num_subsets != 2000:
        config.subsets.num_subsets = args.num_subsets
    if hasattr(args, 'subset_size') and args.subset_size != 8:
        config.subsets.num_random_sequences = args.subset_size
    if hasattr(args, 'num_batches') and args.num_batches != 50:
        config.batches.num_batches = args.num_batches
    if hasattr(args, 'similarity_threshold') and args.similarity_threshold != 0.7:
        config.similarity_search.similarity_threshold = args.similarity_threshold
    
    # Apply skip flags
    config.skip_structure_prediction = args.skip_structure_prediction
    config.skip_structure_analysis = args.skip_structure_analysis
    config.skip_hit_expansion = args.skip_hit_expansion
    
    return config


def load_or_create_config(args: argparse.Namespace) -> PipelineConfig:
    """Load configuration from file or create from CLI arguments."""
    config_manager = ConfigManager()
    
    if args.config_file:
        # Mode 1: Load from existing config file
        config = config_manager.load_config(args.config_file)
        
        # Apply skip flags as overrides
        config.skip_structure_prediction = args.skip_structure_prediction
        config.skip_structure_analysis = args.skip_structure_analysis
        config.skip_hit_expansion = args.skip_hit_expansion
        
        return config
    else:
        # Mode 2: Create from CLI arguments
        return create_config_from_cli(args)


def print_pipeline_summary(config: PipelineConfig) -> None:
    """Print pipeline execution summary."""
    print("\n" + "="*60)
    print("PIPELINE CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"Input file: {config.input_file}")
    print(f"Source MSA file: {config.source_msa_file or 'Not provided'}")
    print(f"Output directory: {config.output_dir}")
    print(f"af_claseq config: {config.structure_analysis.config_file or 'Not provided'}")
    
    print("\nSteps to execute:")
    steps = []
    if not config.skip_structure_prediction:
        steps.append(f"  1. Structure prediction ({config.subsets.num_subsets} subsets)")
    if not config.skip_structure_analysis:
        steps.append(f"  2. Structure analysis (pLDDT>{config.structure_analysis.plddt_threshold})")
    if not config.skip_hit_expansion:
        steps.append(f"  3. Hit expansion (BLOSUM62 similarity>{config.similarity_search.similarity_threshold})")
    
    if steps:
        print("\n".join(steps))
    else:
        print("  No steps to execute (all skipped)")
    
    print("="*60)


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle config creation
    if hasattr(args, 'create_config') and args.create_config:
        config_manager = ConfigManager()
        config = config_manager.create_default_config(args.create_config)
        print(f"Default configuration created: {args.create_config}")
        return
    
    # Determine mode based on arguments
    config_mode = bool(args.config_file)
    cli_mode = bool(args.input_file or args.source_msa or args.output)
    
    if config_mode and cli_mode:
        print("Error: Cannot mix config file mode and CLI mode.")
        print("Use either: python scripts/run_pipeline.py config.yaml")
        print("Or: python scripts/run_pipeline.py --input-file ... --source-msa ... --output ...")
        sys.exit(1)
    
    if not config_mode and not cli_mode:
        print("Error: Must provide either config file or CLI arguments.")
        print("Use: python scripts/run_pipeline.py config.yaml")
        print("Or: python scripts/run_pipeline.py --input-file ... --source-msa ... --output ...")
        print("Use --create-config to generate a default configuration file.")
        sys.exit(1)
    
    if cli_mode:
        # Validate required CLI arguments
        required_args = ['input_file', 'source_msa', 'output']
        missing_args = [arg for arg in required_args if not getattr(args, arg)]
        if missing_args:
            print(f"Error: Missing required arguments in CLI mode: {', '.join(missing_args)}")
            print("Required: --input-file, --source-msa, --output")
            sys.exit(1)
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Load/create configuration
        config = load_or_create_config(args)
        
        # Print mode information
        if args.config_file:
            print(f"\n📄 Running in CONFIG MODE with: {args.config_file}")
        else:
            print(f"\n⚙️  Running in CLI MODE - generating config from arguments")
        
        # Validate inputs from config
        if not config.input_file:
            raise ValueError("Input file must be specified in configuration")
        
        if not Path(config.input_file).exists():
            raise FileNotFoundError(f"Input file not found: {config.input_file}")
        
        # Save config if in CLI mode and save-config is specified
        if cli_mode and args.save_config:
            config_manager = ConfigManager()
            config_manager.save_config(config, args.save_config)
            print(f"Configuration saved to: {args.save_config}")
        
        # Print summary and confirm
        if not args.dry_run:
            print_pipeline_summary(config)
            if not args.yes:
                response = input("\nProceed with pipeline execution? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    print("Pipeline execution cancelled.")
                    return
            else:
                print("\n✅ Auto-proceeding with pipeline execution (--yes flag used)")
        else:
            print_pipeline_summary(config)
            print("\nDry run completed. No files were processed.")
            return
        
        # Run pipeline
        logger.info("Starting MSA optimization pipeline")
        
        pipeline = MSAOptimizationPipeline(config)
        final_output = pipeline.run_full_pipeline(source_msa_file=config.source_msa_file)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📄 Final optimized MSA: {final_output}")
        print(f"📁 Results directory: {config.output_dir}")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        print("\n❌ Pipeline interrupted.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()