# Figure 7 — GB98 (designed fold-switch protein)

Two alternative folds of the designed GB98 sequence: **2LHC** (3α fold) vs **2LHD**
(4β+α fold), scored by Cα-RMSD to each reference.

**Method (hierarchical AF_Vote, two rounds).** A first hierarchical m-fold sampling round
separates the two folds along the 2LHC/2LHD RMSD axes (panel c). Because 2LHD is the rarer
fold, a **second** hierarchical round focuses sampling on the 2LHD region (panel d), after
which sequences vote into bins. Winning bins are recompiled and re-predicted (5 models × 8
seeds = 40 structures): **2LHC bins 4 & 5**, **2LHD bin 5**. Each purified bin ("pure") is
compared against a random-MSA control ("control"). The two `*_values.csv` files (round 1 vs
round 2) are the iterative-enrichment comparison data.

## Panel → file → source provenance

All source files copied (read-only) from `results/supple_cases/GB98/run1/`.

| Panel | File(s) | Source |
|-------|---------|--------|
| c | `panel_c_round1_sampling/{2lhc,2lhd}_rmsd_1d_distribution.png` | `02_hierarchical_m_fold_sampling_plot/{2lhc,2lhd}_rmsd_1d_distribution.png` |
| c (data) | `panel_c_round1_sampling/round1_2lhd_2lhc_rmsd_values.csv` | `02_hierarchical_m_fold_sampling/2lhd_rmsd_2lhc_rmsd_values.csv` |
| d | `panel_d_…/round2_{2lhc,2lhd}_rmsd_1d_distribution.png` | `03_hierarchical_m_fold_sampling_2lhd_plot/{2lhc,2lhd}_rmsd_1d_distribution.png` |
| d (data) | `panel_d_…/round2_2lhd_2lhc_rmsd_values.csv` | `03_hierarchical_m_fold_sampling_2lhd/2lhd_rmsd_2lhc_rmsd_values.csv` |
| d (voting) | `panel_d_…/{2lhc,2lhd}_rmsd_voting_distribution.png` + `…_voting_results.csv` | `03_voting/{2lhc,2lhd}_rmsd/03_voting_distribution.png` + `03_voting_results.csv` |
| f | `panel_f_…/{2lhc_bin4,2lhc_bin5,2lhd_bin5}_{pure,control}_scatter_plddt.png` | `04_plots/{2lhc_rmsd/bin_4,2lhc_rmsd/bin_5,2lhd_rmsd/bin_5}/{prediction,control_prediction}/2lhd_rmsd_2lhc_rmsd_scatter_plddt.png` |
| f (data) | `panel_f_…/*_metrics.csv` (40 rows each) | corresponding `{prediction,control_prediction}_metrics.csv` |
