# Figure 5 — GLP-1 receptor (GLP1R)

Active vs inactive states of the GLP-1 receptor, separated along the **TM3–TM6
intracellular distance** collective variable (with the G361 backbone φ as a secondary
coordinate).

**Method (m-fold sampling → voting).** The MSA is sub-sampled into m-folds, each folded
with ColabFold and scored by the TM3–TM6 distance; sequences vote into bins along that
axis. The winning bins are recompiled and re-predicted (5 models × 8 seeds = 40
structures). **Bin 19 = inactive**, **bin 76 = active**. Each purified bin ("pure") is
compared against a random-MSA control ("control").

## Panel → file → source provenance

All source files copied (read-only) from
`results_updated/GLP1R/run3/m_fold_sampling_voting/`.

| Panel | File(s) in this folder | Source |
|-------|------------------------|--------|
| b | `panel_b_mfold_scatter/TM3_TM6_distance_G361_phi_scatter_plddt.png` | `01_m_fold_sampling/plot/TM3_TM6_distance_G361_phi_scatter_plddt.png` |
| b (data) | `panel_b_mfold_scatter/TM3_TM6_distance_G361_phi_values.csv` | `01_m_fold_sampling/csv/TM3_TM6_distance_G361_phi_values.csv` |
| c | `panel_c_distribution_and_voting/TM3_TM6_distance_1d_distribution.png` + `sequence_voting_distribution.png` | `01_m_fold_sampling/plot/…1d_distribution.png` ; `02_voting/TM3_TM6_distance/sequence_voting_distribution.png` |
| c (data) | `panel_c_distribution_and_voting/{TM3_TM6_distance_values,voting_results}.csv` | `01_m_fold_sampling/csv/TM3_TM6_distance_values.csv` ; `02_voting/TM3_TM6_distance/voting_results.csv` |
| d | `panel_d_…/bin{19_inactive,76_active}_{pure,control}_scatter_plddt.png` | `04_plots/TM3_TM6_distance/bin_{19,76}/{prediction,control_prediction}/…scatter_plddt.png` |
| d (data) | `panel_d_…/*_metrics.csv` (40 rows each) | `04_plots/TM3_TM6_distance/bin_{19,76}/{prediction,control_prediction}_metrics.csv` |
