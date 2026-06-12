# Figure 3 — Adenylate Kinase (KAD_ECOLI)

Two-state separation of *E. coli* adenylate kinase: **1AKE** (closed) vs **4AKE** (open).

**Method (m-fold sampling → voting).** The full MSA is repeatedly sub-sampled into
"m-folds"; each fold is folded with ColabFold and scored by Cα-RMSD to the 1AKE and
4AKE references (9,900 sampled structures). The two RMSD axes are each binned (40 bins);
every sequence votes for the bin it most often lands in; the winning bins are recompiled
and re-predicted (final set = 5 models × 8 seeds = 40 structures). Bin 12 isolates the
1AKE (closed) state, bin 13 the 4AKE (open) state. Each purified bin ("pure") is compared
against a random-MSA control ("control").

## Panel → file → source provenance

All source files copied (read-only) from `results_NC_revision/KAD_ECOLI/`.

| Panel | File in this folder | Source |
|-------|---------------------|--------|
| b | `panel_b_mfold_scatter/1ake_4ake_rmsd_scatter_plddt.png` | `01_m_fold_sampling/plot/1ake_rmsd_4ake_rmsd_scatter_plddt.png` |
| b (data) | `panel_b_mfold_scatter/1ake_4ake_rmsd_values.csv` (9,900 rows) | `01_m_fold_sampling/csv/1ake_rmsd_4ake_rmsd_values.csv` |
| c top | `panel_c_distributions_and_voting/{1ake,4ake}_rmsd_1d_distribution.png` | `01_m_fold_sampling/plot/{1ake,4ake}_rmsd_1d_distribution.png` |
| c top (data) | `panel_c_distributions_and_voting/{1ake,4ake}_rmsd_values.csv` | `01_m_fold_sampling/csv/{1ake,4ake}_rmsd_values.csv` |
| c bottom | `panel_c_distributions_and_voting/{1ake,4ake}_sequence_voting_distribution.png` | `02_voting/{1ake,4ake}_rmsd/sequence_voting_distribution.png` |
| c bottom (data) | `panel_c_distributions_and_voting/{1ake,4ake}_voting_results.csv` | `02_voting/{1ake,4ake}_rmsd/voting_results.csv` |
| d top (pure) | `panel_d_pure_vs_control_predictions/{1ake_bin12,4ake_bin13}_pure_scatter_plddt.png` | `04_plots/1ake_rmsd/bin_12/prediction/…` and `04_plots/4ake_rmsd/bin_13/prediction/…` |
| d bottom (control) | `panel_d_pure_vs_control_predictions/{1ake_bin12,4ake_bin13}_control_scatter_plddt.png` | `04_plots/…/control_prediction/…` |
| d (data) | `panel_d_pure_vs_control_predictions/*_metrics.csv` (40 rows each) | `04_plots/…/bin_*/{prediction,control_prediction}_metrics.csv` |

`composed_figure_3.png` — the assembled manuscript panel (`composed_figure.png`).
