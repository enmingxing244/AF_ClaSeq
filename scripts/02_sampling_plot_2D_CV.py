import argparse
from af_claseq.plotting_manager import plot_m_fold_sampling_2d, plot_m_fold_sampling_2d_joint
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Generate plots for structure analysis metrics')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to directory containing results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to directory for saving plot outputs')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Path to directory for saving CSV results')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--x_min', type=float, default=None,
                        help='Minimum x-axis value')
    parser.add_argument('--x_max', type=float, default=None,
                        help='Maximum x-axis value')
    parser.add_argument('--y_min', type=float, default=None,
                        help='Minimum y-axis value')
    parser.add_argument('--y_max', type=float, default=None,
                        help='Maximum y-axis value')
    parser.add_argument('--plddt_threshold', type=float, default=0,
                        help='pLDDT threshold for filtering structures (default: 0, no filtering)')
    parser.add_argument('--x_ticks', nargs='*', type=float, default=None,
                        help='List of x-axis tick values')
    parser.add_argument('--y_ticks', nargs='*', type=float, default=None,
                        help='List of y-axis tick values')
    
    args = parser.parse_args()
    
    plot_m_fold_sampling_2d(
        results_dir=args.results_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        csv_dir=args.csv_dir,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        plddt_threshold=args.plddt_threshold,
        x_ticks=args.x_ticks,
        y_ticks=args.y_ticks
    )

    plot_m_fold_sampling_2d_joint(
        results_dir=args.results_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        csv_dir=args.csv_dir,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        plddt_threshold=args.plddt_threshold
    )


if __name__ == "__main__":
    main()
