import argparse
from af_claseq.plotting_manager import plot_m_fold_sampling_1d
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color


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
    parser.add_argument('--initial_color', type=str, default='#87CEEB',
                        help='Color in hex format (e.g. #d5f6dd, default: #87CEEB skyblue)')
    parser.add_argument('--end_color', type=str, default='#FFFFFF',
                        help='Color in hex format (e.g. #d5f6dd, default: #FFFFFF white)')
    parser.add_argument('--x_min', type=float, default=None,
                        help='Minimum x-axis value')
    parser.add_argument('--x_max', type=float, default=None,
                        help='Maximum x-axis value')
    parser.add_argument('--log_scale', action='store_true',
                        help='Use log scale for y-axis in distribution plots')
    parser.add_argument('--n_plot_bins', type=int, default=50,
                        help='Number of bins for histogram (default: 50)')
    parser.add_argument('--iteration_dirs', nargs='*', default=None,
                        help='List of iteration directories to process (optional)')
    parser.add_argument('--gradient_ascending', action='store_true',
                        help='Use ascending gradient for color')
    parser.add_argument('--linear_gradient', action='store_true',
                        help='Use linear gradient for color')
    parser.add_argument('--plddt_threshold', type=float, default=0,
                        help='pLDDT threshold for filtering structures (default: 0, no filtering)')
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 5),
                        help='Figure size in inches (width, height) (default: 10 5)')
    parser.add_argument('--show_bin_lines', action='store_true', default=False,
                        help='Show vertical dashed lines at bin boundaries (default: False) ')
    parser.add_argument('--y_min', type=float, default=None,
                        help='Minimum y-axis value')
    parser.add_argument('--y_max', type=float, default=None,
                        help='Maximum y-axis value')
    parser.add_argument('--x_ticks', nargs='*', type=float, default=None,
                        help='List of x-axis tick values (optional)')
    
    args = parser.parse_args()

    print(args.gradient_ascending)
    
    plot_m_fold_sampling_1d(
        results_dir=args.results_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        csv_dir=args.csv_dir,
        initial_color=hex2color(args.initial_color),
        end_color=hex2color(args.end_color),
        x_min=args.x_min,
        x_max=args.x_max,
        log_scale=args.log_scale,
        n_plot_bins=args.n_plot_bins,
        gradient_ascending=args.gradient_ascending,
        linear_gradient=args.linear_gradient,
        plddt_threshold=args.plddt_threshold,
        figsize=args.figsize,
        show_bin_lines=args.show_bin_lines,
        y_min=args.y_min,
        y_max=args.y_max,
        x_ticks=args.x_ticks
    )
     
if __name__ == "__main__":
    main()
