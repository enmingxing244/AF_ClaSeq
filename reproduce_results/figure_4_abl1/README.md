# Figure 4 — ABL1 kinase

Two-state separation of ABL1: **6XR6** (active) vs **6XRG** (inactive), scored by a
composite A-loop Cα-RMSD.

**Method (two rounds of m-fold sampling).** Round 1 ("random") runs m-fold sampling on
the full MSA; the active state (6XR6) separates cleanly but the inactive state (6XRG) is
under-sampled. To recover it, clade-based divide-and-conquer + occurrence-voting enrichment
selects an inactive-biased sequence set, which is fed into a **second** round of m-fold
sampling ("enriched"). Winning bins are recompiled and re-predicted (5 models × 8 seeds =
40 structures). 6XR6 active = bin 10 (round 1); 6XRG inactive = bin 16 (round 2). The
inactive control is 22 random sequences (`random_22_seqs`).

## Panel → file → source provenance

All source files copied (read-only) from `results_NC_revision/ABL1/`.
`round1_random` = `m_fold_random/`; `round2_enriched` = `m_fold_occur_voted_top50/`.

| Panel | File(s) in this folder | Source |
|-------|------------------------|--------|
| b | `panel_b_mfold_distributions/round1_random_{6xr6_active,6xrg_inactive}_1d_distribution.png` | `m_fold_random/01_m_fold_sampling/plot/{6xr6,6xrg}_composite_rmsd_1d_distribution.png` |
| b | `panel_b_mfold_distributions/round2_enriched_6xrg_inactive_1d_distribution.png` | `m_fold_occur_voted_top50/01_m_fold_sampling/plot/6xrg_composite_rmsd_1d_distribution.png` |
| b (data) | `…_values.csv` (×3) | corresponding `01_m_fold_sampling/csv/{6xr6,6xrg}_composite_rmsd_values.csv` |
| c | `panel_c_voting/*_voting_distribution.png` (×3) | `02_voting/{6xr6,6xrg}_composite_rmsd/sequence_voting_distribution.png` (both rounds) |
| c (data) | `panel_c_voting/*_voting_results.csv` (×3) | `02_voting/…/voting_results.csv` |
| d | `panel_d_…/6xr6_active_bin10_{pure,control}_scatter_plddt.png` | `m_fold_random/04_plots/6xr6_composite_rmsd/bin_10/{prediction,control_prediction}/…scatter_plddt.png` |
| d | `panel_d_…/6xrg_inactive_bin16_pure_scatter_plddt.png` | `m_fold_occur_voted_top50/04_plots/6xrg_composite_rmsd/bin_16/prediction/…` |
| d | `panel_d_…/6xrg_inactive_random22_control_scatter_plddt.png` | `m_fold_occur_voted_top50/random_22_seqs/…` |
| d (data) | `panel_d_…/*_metrics.csv` (40 rows each) | corresponding `{prediction,control_prediction}_metrics.csv` |
