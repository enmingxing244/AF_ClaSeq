# Figure 6 — KaiB

Fold-switching of KaiB between its **ground state (2QKE)** and the rare **fold-switched
state (5JYT)**, scored by TM-score to each reference.

**Method (m-fold sampling → voting, one run per state).** After a divide-and-conquer
pre-pass on the MSA, two independent m-fold sampling+voting runs are carried out — one
voting on TM-score to 5JYT (fold-switch), one on TM-score to 2QKE (ground). Winning bins
are recompiled and re-predicted (5 models × 8 seeds = 40 structures): **5JYT bin 41**
(fold-switch), **2QKE bin 51** (ground). Each purified bin ("pure") is compared against a
random-MSA control ("control"). The joint TM-score scatter (2QKE vs 5JYT) shows the two
states as anti-correlated clusters.

## Panel → file → source provenance

All source files copied (read-only) from
`results_updated/KaiB/run1/m_fold_sampling_voting_{5jyt,2qke}/`.

| Panel | File(s) | Source |
|-------|---------|--------|
| b (joint) | `panel_b_…/joint_2qke_5jyt_tmscore_scatter_plddt.png` + `…_values.csv` | `…_5jyt/01_m_fold_sampling/{plot,csv}/2qke_tmscore_5jyt_tmscore_*` |
| b (fold-switch) | `panel_b_…/foldswitch_5jyt_tmscore_1d_distribution.png`, `foldswitch_5jyt_voting_distribution.png` (+ CSVs) | `…_5jyt/01_m_fold_sampling/…/5jyt_tmscore_*` ; `…_5jyt/02_voting/5jyt_tmscore/*` |
| b (ground) | `panel_b_…/ground_2qke_tmscore_1d_distribution.png`, `ground_2qke_voting_distribution.png` (+ CSVs) | `…_2qke/01_m_fold_sampling/…/2qke_tmscore_*` ; `…_2qke/02_voting/2qke_tmscore/*` |
| c | `panel_c_…/{foldswitch_5jyt_bin41,ground_2qke_bin51}_{pure,control}_scatter_plddt.png` | `…/04_plots/{5jyt_tmscore/bin_41,2qke_tmscore/bin_51}/{prediction,control_prediction}/…` |
| c (data) | `panel_c_…/*_metrics.csv` (40 rows each) | corresponding `{prediction,control_prediction}_metrics.csv` |
