# Supplementary Figure 1 — AtaR toxin (AtaTR)

Two conformational states of AtaTR, scored by Cα-RMSD to references **6AJM** and **6GTO**.

**Method (hierarchical AF_Vote).** Hierarchical m-fold sampling separates the two states
along the 6AJM/6GTO RMSD axes; sequences vote into bins; the winning bins are recompiled
and re-predicted (5 models × 8 seeds = 40 structures): **6AJM bin 8**, **6GTO bin 3**.
Each purified bin ("pure") is compared against a random-MSA control ("control").

## Panel → file → source provenance

All source files copied (read-only) from `results/supple_cases/AtaTR/run1/`. (RMSD axes
used for the figure; the parallel `*_tmscore_*` files in the source are not plotted here.)

| Panel | File(s) | Source |
|-------|---------|--------|
| b | `panel_b_…/{6ajm,6gto}_rmsd_1d_distribution.png` | `02_hierarchical_m_fold_sampling_plot/{6ajm,6gto}_rmsd_1d_distribution.png` |
| b (data) | `panel_b_…/6ajm_6gto_rmsd_values.csv` | `02_hierarchical_m_fold_sampling_plot/6ajm_rmsd_6gto_rmsd_values.csv` |
| b (voting) | `panel_b_…/{6ajm,6gto}_rmsd_voting_distribution.png` + `…_voting_results.csv` | `03_voting/{6ajm,6gto}_rmsd/03_voting_distribution.png` + `03_voting_results.csv` |
| d | `panel_d_…/{6ajm_bin8,6gto_bin3}_{pure,control}_scatter_plddt.png` | `05_plot/{6ajm_rmsd/bin_8,6gto_rmsd/bin_3}/{prediction,control_prediction}_metric_correlation.png` |
| d (data) | `panel_d_…/*_metrics.csv` (40 rows each) | corresponding `{prediction,control_prediction}_metrics.csv` |
