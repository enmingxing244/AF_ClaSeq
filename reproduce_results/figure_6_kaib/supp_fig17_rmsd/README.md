# Supplementary Figure 17 — KaiB RMSD analysis

A supplement to Figure 6. Whereas Figure 6 scores KaiB by **TM-score** to each reference,
this analysis repeats the separation using **Cα-RMSD** to 5JYT (fold-switch) and 2QKE
(ground), confirming the two states are recovered independently of the metric.

**Method (RMSD-based m-fold sampling → voting, one run per state).** Two independent
RMSD-scored m-fold sampling+voting runs (5JYT and 2QKE). Winning bins are recompiled and
re-predicted (5 models × 8 seeds = 40 structures): **5JYT bin 11** (fold-switch), **2QKE
bin 8** (ground). Each purified bin ("pure") vs a random-MSA control ("control").

## File → source provenance

All sources copied (read-only) from
`results_NC_revision/KaiB/m_fold_sampling_voting_{5jyt,2qke}/`.

| Content | File | Source |
|---------|------|--------|
| Joint 2-D RMSD scatter | `sampling_and_voting/joint_2qke_5jyt_rmsd_scatter_plddt.png` + `…_values.csv` | `…_5jyt/01_m_fold_sampling/{plot,csv}/2qke_rmsd_5jyt_rmsd_*` |
| Fold-switch (5JYT) dist + voting | `sampling_and_voting/foldswitch_5jyt_rmsd_1d_distribution.png`, `foldswitch_5jyt_voting_distribution.png` (+ CSVs) | `…_5jyt/01_m_fold_sampling/…/5jyt_rmsd_*` ; `…_5jyt/02_voting/5jyt_rmsd/*` |
| Ground (2QKE) dist + voting | `sampling_and_voting/ground_2qke_rmsd_1d_distribution.png`, `ground_2qke_voting_distribution.png` (+ CSVs) | `…_2qke/01_m_fold_sampling/…/2qke_rmsd_*` ; `…_2qke/02_voting/2qke_rmsd/*` |
| Predictions | `pure_vs_control_predictions/{foldswitch_5jyt_bin11,ground_2qke_bin8}_{pure,control}_scatter_plddt.png` (+ `_metrics.csv`, 40 rows) | `…/04_plots/{5jyt_rmsd/bin_11,2qke_rmsd/bin_8}/{prediction,control_prediction}/…` |

(Folder named by the manuscript supplementary-figure number; confirm the exact number
against the final supplement.)
