# Supplementary Figure 2 — human H-Ras (RASH_HUMAN)

Active vs inactive states of H-Ras: **5P21** (active, GppNHp-bound) vs **4Q21** (inactive,
GDP-bound), scored by Cα-RMSD to each reference.

**Method (iterative enrichment → hierarchical AF_Vote).** The ~53,670-sequence DeepMSA2 MSA
is first refined by **10 iterations of iterative shuffling** to enrich state-discriminating
sequences. Hierarchical m-fold sampling then separates the two states along the 5P21/4Q21
RMSD axes; sequences vote into bins; winning bins are recompiled and re-predicted (5 models
× 8 seeds = 40 structures): **5P21 (active) bins 6–7**, **4Q21 (inactive) bins 7–9**. Each
purified bin ("pure") is compared against a random-MSA control ("control").

## Panel → file → source provenance

All source files copied (read-only) from `results/supple_cases/RASH_HUMAN/run1/`.

| Panel | File(s) | Source |
|-------|---------|--------|
| c | `panel_c_…/{5p21_active,4q21_inactive}_rmsd_1d_distribution.png` | `02_hierarchical_m_fold_sampling_plot/{5p21,4q21}_rmsd_1d_distribution.png` |
| c (data) | `panel_c_…/4q21_5p21_rmsd_values.csv` | `02_hierarchical_m_fold_sampling_plot/4q21_rmsd_5p21_rmsd_values.csv` |
| c (voting) | `panel_c_…/{5p21_active,4q21_inactive}_voting_distribution.png` + `…_voting_results.csv` | `03_voting/{5p21,4q21}_rmsd/03_voting_distribution.png` + `03_voting_results.csv` |
| e | `panel_e_…/{5p21_active_bin6-7,4q21_inactive_bin7-9}_{pure,control}_scatter_plddt.png` | `05_plot/{5p21_rmsd/bin_6_7,4q21_rmsd/bin_7_8_9}/{prediction,control_prediction}_metric_correlation.png` |
| e (data) | `panel_e_…/*_metrics.csv` (40 rows each) | corresponding `{prediction,control_prediction}_metrics.csv` |

The 10-iteration iterative-shuffling intermediates (`01_iterative_shuffling/`) are a
preprocessing step and are not deposited here.
