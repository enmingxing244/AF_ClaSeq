#!/usr/bin/env bash
#
# reproduce_figures.sh — regenerate every KaiB demo figure from the
# pre-computed ColabFold predictions. No GPU, no job submission.
#
# Before running:
#   1. Download the pre-computed results tarball from the Google Drive link
#      in README.md and place it in THIS folder (example/KaiB_demo/).
#   2. Extract it here so that ./m_fold_sampling_voting/ exists:
#         tar -xzf m_fold_sampling_voting.tar.gz
#
# Then just run:
#         bash reproduce_figures.sh
#
# It re-scores the predicted structures with TM-align and regenerates the
# analysis, voting, and final figures under ./m_fold_sampling_voting/.

set -euo pipefail

# Always run from the folder this script lives in, so the relative paths
# in reproduce_figures.yaml (source_a3m, config_file, ref/, base_dir) resolve.
cd "$(dirname "$0")"

PRECOMPUTE_DIR="m_fold_sampling_voting"
CONFIG="reproduce_figures.yaml"
DRIVER="../../scripts/run_m_fold_sampling_voting.py"

if [[ ! -d "$PRECOMPUTE_DIR" ]]; then
  echo "ERROR: '$PRECOMPUTE_DIR/' not found in $(pwd)"
  echo
  echo "Download the pre-computed results tarball (Google Drive link in README.md),"
  echo "put it in this folder, then extract it here:"
  echo "    tar -xzf m_fold_sampling_voting.tar.gz"
  echo
  echo "After extraction this folder should contain '$PRECOMPUTE_DIR/'. Re-run this script."
  exit 1
fi

echo "Reproducing KaiB demo figures from pre-computed results in ./$PRECOMPUTE_DIR/ ..."
python "$DRIVER" "$CONFIG"

echo
echo "Done. Regenerated figures live under ./$PRECOMPUTE_DIR/ :"
echo "  01_m_fold_sampling/plot/   landscape histograms + 2D TM-score scatter"
echo "  02_voting/<metric>/        per-metric sequence-voting distributions"
echo "  04_plots/<metric>/bin_*/   final purified-bin scatters (vs. random control)"
