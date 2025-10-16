#!/bin/bash

# Batch processing script for all phylogenetic trees
# Each case is explicitly written out for easy modification

PLOT_SCRIPT="/Users/enmingxing/Projects/AF_ClaSeq_NSMB_revision/tree_analysis_plot/tree/plot_tree_circular.py"
BASE_DIR="/Users/enmingxing/Projects/AF_ClaSeq_NSMB_revision/tree_analysis_plot/tree/official_plot/all_tree"

# ======================================
# ABL1
# ======================================
echo "Processing ABL1..."
cd "${BASE_DIR}/ABL1"

python ${PLOT_SCRIPT} \
    -t ABL1_P00519_preprocessed.nwk \
    -o ABL1_tree_plot.png \
    --highlight bin10_sequences_corrected.a3m bin7_sequences_corrected.a3m \
    --no-collapse \
    --no-labels \
    --highlight-size 200

echo "ABL1 done!"
echo ""

# ======================================
# GB98
# ======================================
echo "Processing GB98..."
cd "${BASE_DIR}/GB98"

python ${PLOT_SCRIPT} \
    -t GB98_1-56_preprocessed.nwk \
    -o GB98_tree_plot.png \
    --highlight 2lhc_bin4_bin5_sequences_corrected.a3m 2lhd_bin5_sequences.a3m \
    --no-collapse \
    --no-labels \
    --highlight-size 200

echo "GB98 done!"
echo ""

# ======================================
# GLP1R
# ======================================
echo "Processing GLP1R..."
cd "${BASE_DIR}/GLP1R"

python ${PLOT_SCRIPT} \
    -t GLP1R_P43220_preprocessed.nwk \
    -o GLP1R_tree_plot.png \
    --highlight bin19_sequences.a3m bin76_sequences.a3m \
    --no-collapse \
    --no-labels \
    --highlight-size 200

echo "GLP1R done!"
echo ""

# ======================================
# KAD_ECOLI
# ======================================
# echo "Processing KAD_ECOLI..."
# cd "${BASE_DIR}/KAD_ECOLI"

# python ${PLOT_SCRIPT} \
#     -t KAD_ECOLI_preprocessed.nwk \
#     -o KAD_ECOLI_tree_plot.png \
#     --highlight  bin21_sequences_corrected.a3m bin19_sequences_corrected.a3m \
#     --no-collapse \
#     --no-labels \
#     --highlight-size 200

# echo "KAD_ECOLI done!"
# echo ""

# ======================================
# KaiB
# ======================================
# echo "Processing KaiB..."
# cd "${BASE_DIR}/KaiB"

# python ${PLOT_SCRIPT} \
#     -t KAIB_THEVB_Q79V61_5-95_preprocessed.nwk \
#     -o KaiB_tree_plot.png \
#     --highlight bin41_sequences_corrected.a3m bin51_sequences_corrected.a3m \
#     --n-clades 5 \
#     --no-labels \
#     --highlight-size 200

# echo "KaiB done!"
# echo ""

# ======================================
echo "All cases processed!"
echo "Output files:"
echo "  - ${BASE_DIR}/ABL1/ABL1_tree_plot.png (and .pdf)"
echo "  - ${BASE_DIR}/GB98/GB98_tree_plot.png (and .pdf)"
echo "  - ${BASE_DIR}/GLP1R/GLP1R_tree_plot.png (and .pdf)"
echo "  - ${BASE_DIR}/KAD_ECOLI/KAD_ECOLI_tree_plot.png (and .pdf)"
echo "  - ${BASE_DIR}/KaiB/KaiB_tree_plot.png (and .pdf)"
