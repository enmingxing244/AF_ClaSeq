# KaiB demo — reproducing the two-fold separation from pre-computed results

KaiB is a metamorphic protein that natively adopts two completely different folds: a **ground state** (PDB **2QKE**) and a **fold-switch state** (PDB **5JYT**). The **M-Fold Sampling & Voting** workflow scores ColabFold predictions of a 349-sequence KaiB MSA by **TM-score to each reference**, then votes to pull apart which sequences encode which fold.

This demo lets you **reproduce all of that from a set of pre-computed ColabFold predictions** — no GPU and no job submission. You download the predicted structures, run one command, and the analysis, voting, and final figures are regenerated on your machine in a couple of minutes.

## What's in this folder

| File | Role |
|------|------|
| `reproduce_figures.sh` | **Run this** — one command (`bash reproduce_figures.sh`) that regenerates every figure from the downloaded predictions. No GPU. |
| `reproduce_figures.yaml` | The config it uses — the figure stages only (analysis → voting → final plots). |
| `KaiB_demo_raw_MSA.a3m` | Input MSA — 349 sequences, 91-residue query. Maps sequences to their voted bins during reproduction. |
| `ref/2qkeE.pdb` | Reference structure for the **ground state** (2QKE, chain E). |
| `ref/5jytA.pdb` | Reference structure for the **fold-switch state** (5JYT, chain A). |
| `configs/config_2qke_5jyt_tmscore.json` | Structure-analysis config: the two TM-score metrics (`2qke_tmscore`, `5jyt_tmscore`), each `above`-type against its reference PDB, over residues 1–91. |
| `KaiB_m_fold_voting_demo.yaml` | The full pipeline config (all stages, incl. structure prediction) that *generated* the pre-computed results — kept for reference. |
| `KaiB_mfold_demo.ipynb` | Optional interactive notebook walkthrough of the same analysis. |

## Prerequisites

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

This pulls in the Python dependencies (biopython, numpy, pandas, matplotlib, seaborn, pyyaml, ete3). You also need **TM-align** on your `PATH` — it scores each predicted structure against the two reference PDBs. **ColabFold and a GPU are _not_ required** for this reproduce flow, since the structures are already predicted.

## Reproduce the figures

**1. Download the pre-computed results** — [KaiB demo pre-computed results](https://drive.google.com/file/d/1XcMJ-yIbOlk7CoSmMC9ewydrBifhhVBc/view?usp=share_link) *(tarball)* — and extract it **inside this folder**:

```bash
# put the downloaded m_fold_sampling_voting.tar.gz into example/KaiB_demo/, then:
cd example/KaiB_demo
tar -xzf m_fold_sampling_voting.tar.gz
```

This creates `m_fold_sampling_voting/`, holding the predicted structures — the per-group ColabFold outputs under `01_m_fold_sampling/round_1/` and the per-bin predictions under `03_recompile/`.

**2. Run the reproduce script** — from inside `example/KaiB_demo/`:

```bash
bash reproduce_figures.sh
```

That's it. The script checks that `m_fold_sampling_voting/` is extracted, then runs the figure-only pipeline (`python ../../scripts/run_m_fold_sampling_voting.py reproduce_figures.yaml`) for you. It re-reads the predicted structures, re-scores them with TM-align, and regenerates every analysis figure into `m_fold_sampling_voting/`. Only three figure stages run — `01_M_FOLD_SAMPLING_PLOT`, `02_VOTING_RUN`, `04_PURE_SEQ_PLOT_RUN` — so there is no prediction and no SLURM/GPU involved.

> ⚠️ The script `cd`s into its own folder, so it works from anywhere. If you call the Python driver directly instead, run it from `example/KaiB_demo/` — the config resolves `source_a3m`, `config_file`, `ref/*.pdb`, and `base_dir` against your current working directory.

## What you get

```
m_fold_sampling_voting/
├── 01_m_fold_sampling/
│   ├── round_1/                         # pre-computed per-group ColabFold predictions (from the tarball)
│   ├── csv/                             # 2qke / 5jyt TM-score values  (regenerated)
│   └── plot/                            # *_1d_distribution, *_scatter_plddt, *_joint_plddt  (regenerated)
├── 02_voting/
│   ├── 2qke_tmscore/                    # voting_results.csv + sequence_voting_distribution.{png,pdf}  (regenerated)
│   └── 5jyt_tmscore/                    # voting_results.csv + sequence_voting_distribution.{png,pdf}  (regenerated)
├── 03_recompile/                        # pre-computed per-bin predictions (from the tarball)
│   ├── 2qke_tmscore/{prediction,control_prediction}/
│   └── 5jyt_tmscore/{prediction,control_prediction}/
├── 04_plots/
│   ├── 2qke_tmscore/bin_51_52/          # prediction_metrics.csv + final scatter  (regenerated)
│   └── 5jyt_tmscore/bin_37_38_39_40_41/ # prediction_metrics.csv + final scatter  (regenerated)
└── logs/af_claseq_pipeline.log
```

How to read them:

- **`01_m_fold_sampling/plot/` histograms & scatter** — because KaiB is bimodal, you should see populations near high `2qke_tmscore` (ground-fold-like) and, separately, near high `5jyt_tmscore` (fold-switch-like).
- **`02_voting/<metric>/voting_results.csv`** — bins are ranked by occupancy; the **high-count bins are the signal**. Sequences dominating the high-`2qke_tmscore` bins encode the **2QKE ground fold**; those dominating the high-`5jyt_tmscore` bins encode the **5JYT fold-switch state**. The bins selected for recompilation (`[51, 52]` for 2QKE, `[37–41]` for 5JYT) come straight from these results.
- **`04_plots/<metric>/bin_*/`** — the final `prediction_metrics.csv` (and its `control_prediction_metrics.csv`) are the payoff: the voted/purified sequences cluster tightly at high TM-score to their target reference, well separated from the random control, confirming each sequence set reconstructs its intended fold.

## Interactive notebook (optional)

`KaiB_mfold_demo.ipynb` walks through the same analysis cell by cell with the figures shown inline.

## More

- Full workflow guide: [../../docs/m_fold_sampling_voting.md](../../docs/m_fold_sampling_voting.md) (and the **M-Fold Sampling & Voting** section of the root [../../README.md](../../README.md)).
