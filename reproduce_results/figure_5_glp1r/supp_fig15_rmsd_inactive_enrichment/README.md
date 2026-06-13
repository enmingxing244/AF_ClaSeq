# Supplementary Figure 15 — GLP1R RMSD analysis & inactive-state enrichment

A supplement to Figure 5. Whereas Figure 5 separates GLP1R along the TM3–TM6 distance
collective variable, this analysis scores by **whole-protein (global) Cα-RMSD** to the
active and inactive references, and shows that an **inactive-enrichment** second round is
needed to recover the under-sampled inactive state.

**Method (two rounds of RMSD-based m-fold sampling).**
- **Round 1 (`round1_mfold_sampling/`)** — standard m-fold sampling on the MSA, scored by
  `active_rmsd_global` / `inactive_rmsd_global`. The active state separates but the inactive
  state is sparsely sampled.
- **Round 2 (`round2_inactive_enriched/`)** — m-fold sampling on an inactive-enriched
  sequence set, which fills in the inactive cluster.

Each round provides the 2-D RMSD scatter, the per-state 1-D distributions, and the
sequence-voting distributions (with backing CSVs).

## File → source provenance

All sources copied (read-only) from `results_NC_revision/GLP1R/` —
`round1` = `m-fold-sampling/`, `round2` = `inactive_enrich_m_fold_sampling/`. The `_global`
RMSD metric (common to both rounds) is used.

| Content | File (each round dir) | Source (`…/<round>/`) |
|---------|-----------------------|------------------------|
| 2-D RMSD scatter | `inactive_active_rmsd_global_scatter_plddt.png` + `…_values.csv` | `01_m_fold_sampling/{plot,csv}/inactive_rmsd_global_active_rmsd_global_*` |
| 1-D distributions | `{active,inactive}_rmsd_global_1d_distribution.png` + `…_values.csv` | `01_m_fold_sampling/{plot,csv}/{active,inactive}_rmsd_global_*` |
| Voting | `{active,inactive}_rmsd_global_voting_distribution.png` + `…_voting_results.csv` | `02_voting/{active,inactive}_rmsd_global/{sequence_voting_distribution.png,voting_results.csv}` |

(Folder named by the manuscript supplementary-figure number; confirm the exact number
against the final supplement.)
