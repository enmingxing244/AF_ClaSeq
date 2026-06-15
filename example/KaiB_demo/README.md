# KaiB demo — separating two folds with M-Fold Sampling & Voting

KaiB is a metamorphic protein that natively adopts two completely different folds: a **ground state** (PDB **2QKE**) and a **fold-switch state** (PDB **5JYT**). This demo runs the **M-Fold Sampling & Voting** workflow on a 349-sequence KaiB MSA, scoring every prediction by **TM-score to each reference** so the pipeline can pull apart which sequences encode which fold — a small, fast end-to-end run that is the ideal first thing to try.

## What's in this folder

| File | Role |
|------|------|
| `KaiB_demo_raw_MSA.a3m` | Input MSA — 349 sequences, 91-residue query. Serves as both the sampling pool and the source for query extraction. |
| `ref/2qkeE.pdb` | Reference structure for the **ground state** (2QKE, chain E). |
| `ref/5jytA.pdb` | Reference structure for the **fold-switch state** (5JYT, chain A). |
| `configs/config_2qke_5jyt_tmscore.json` | Structure-analysis config: defines the two TM-score metrics (`2qke_tmscore`, `5jyt_tmscore`), each `above`-type against its reference PDB, over residues 1–91. |
| `KaiB_m_fold_voting_demo.yaml` | The workflow config that drives all five pipeline stages (paths, bin/plot settings, SLURM settings, selected bins). |
| `KaiB_mfold_demo.ipynb` | Optional interactive notebook that runs the same pipeline on a local multi-GPU box (no SLURM). |

## Don't want to run ColabFold?

The demo can ship with a full set of **pre-computed results** so you can explore the analysis and plots without any GPU work. Note that both the results directory (`example/KaiB_demo/m_fold_sampling_voting/`) and the archive `m_fold_sampling_voting.tar.gz` are **`.gitignore`d** to keep the repository light — they are **not** tracked in git.

- **Download** the tarball — [KaiB demo pre-computed results](https://drive.google.com/file/d/1XcMJ-yIbOlk7CoSmMC9ewydrBifhhVBc/view?usp=share_link) *(tarball archive)* — then extract it in this folder:

  ```bash
  # put the downloaded m_fold_sampling_voting.tar.gz into example/KaiB_demo/, then:
  cd example/KaiB_demo
  tar -xzf m_fold_sampling_voting.tar.gz
  ```

- Once extracted, browse straight to the outputs — e.g. open `m_fold_sampling_voting/01_m_fold_sampling/plot/combined_mfold_analysis.png`, inspect `m_fold_sampling_voting/02_voting/2qke_tmscore/voting_results.csv`, or view the final scatters under `m_fold_sampling_voting/04_plots/`.

If you have neither the directory nor the tarball, regenerate everything by running the pipeline as described in **Run the demo** below.

## Prerequisites

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

This pulls in all Python dependencies (biopython, numpy, pandas, matplotlib, seaborn, pyyaml, ete3). You also need **ColabFold** installed and configured with **GPU access** for the structure-prediction stages. See the install section of the root README for the full prerequisite list and setup steps: [../../README.md](../../README.md).

> **FastTree** and **TM-align** are external tools used by AF_ClaSeq. FastTree builds phylogenetic trees and is only needed for the tree-based **Divide-and-Conquer** workflow — this M-Fold demo does **not** require it. TM-align (TM-score) is what scores each prediction against the two reference PDBs, so it does need to be available.

## Run the demo

The YAML uses paths **relative to the demo folder** (e.g. `source_a3m: "KaiB_demo_raw_MSA.a3m"`, `config_file: "configs/config_2qke_5jyt_tmscore.json"`, `base_dir: "m_fold_sampling_voting"`), so you must launch it from **inside this folder**:

```bash
cd example/KaiB_demo
python ../../scripts/run_m_fold_sampling_voting.py KaiB_m_fold_voting_demo.yaml
```

> ⚠️ Run it from `example/KaiB_demo/`, **not** the repo root. The pipeline resolves `source_a3m`, `config_file`, `ref/*.pdb`, and `base_dir` against your *current working directory*, so launching from anywhere else fails right away with `ValueError: JSON config file not found: configs/config_2qke_5jyt_tmscore.json`.
>
> ⚠️ Before the prediction stages can submit jobs, edit the three SLURM placeholders in the YAML — `conda_env_path`, `slurm_account`, `slurm_partition` — to your cluster's values (see [SLURM vs local](#slurm-vs-local) below).

### The five stages

This workflow does **not** take per-stage command-line flags. Instead, the script runs whichever stages are listed under `pipeline_control.stages` in the YAML. The demo config enables all five in order:

```yaml
pipeline_control:
  stages:
    - "01_M_FOLD_SAMPLING_RUN"    # 1. Sampling — split MSA into groups of 10, predict each with ColabFold
    - "01_M_FOLD_SAMPLING_PLOT"   # 2. Analysis & plotting — TM-score every prediction, bin, histogram + scatter
    - "02_VOTING_RUN"             # 3. Sequence voting — count which sequences land in the top bins per metric
    - "03_RECOMPILE_PREDICT_RUN"  # 4. Predict selected bins — re-predict voted bins with 5 models × 8 seeds
    - "04_PURE_SEQ_PLOT_RUN"      # 5. Final plots — prediction vs. control scatter for each metric
```

To run stages **individually** (recommended the first time, since stage 1 must finish before stage 2 has data), comment out the stages you don't want yet and re-run the same command — completed stages leave their outputs on disk and are simply skipped on the next invocation:

```bash
# From example/KaiB_demo/ — edit KaiB_m_fold_voting_demo.yaml so only "01_M_FOLD_SAMPLING_RUN" is uncommented, then:
python ../../scripts/run_m_fold_sampling_voting.py KaiB_m_fold_voting_demo.yaml

# When that finishes, uncomment "01_M_FOLD_SAMPLING_PLOT" and re-run, and so on.
```

What each stage does, in detail:

1. **`01_M_FOLD_SAMPLING_RUN` — sampling.** Splits the 349-sequence MSA into groups of `m_fold_group_size: 10` for `rounds: 1` round, then submits a ColabFold prediction for each group. Outputs land in `m_fold_sampling_voting/01_m_fold_sampling/round_1/`.
2. **`01_M_FOLD_SAMPLING_PLOT` — analysis & plotting.** Computes the `2qke_tmscore` and `5jyt_tmscore` for every prediction, bins them, and writes per-metric 1D histograms, a 2D joint plot, and a scatter, plus the underlying CSVs.
3. **`02_VOTING_RUN` — sequence voting.** For each metric, finds the high-scoring bins and counts how often each sequence appears in them, producing a per-metric `voting_results.csv` and a voting-distribution plot.
4. **`03_RECOMPILE_PREDICT_RUN` — predict selected bins.** Recompiles the sequences from the bins chosen in the YAML (`bin_numbers_1: [51, 52]` for `2qke_tmscore`, `bin_numbers_2: [37, 38, 39, 40, 41]` for `5jyt_tmscore`) and re-predicts each with `prediction_num_model: 5` models × `prediction_num_seed: 8` seeds for a robust structural readout. A matched random-sequence **control** is predicted alongside.
5. **`04_PURE_SEQ_PLOT_RUN` — final plots.** Generates the prediction-vs-control TM-score scatter plots that show the purified bins collapsing onto the correct fold.

## Expected outputs

Everything is written under the `base_dir` set in the YAML — here, `example/KaiB_demo/m_fold_sampling_voting/`:

```
m_fold_sampling_voting/
├── 01_m_fold_sampling/
│   ├── round_1/                         # per-group sampling + predictions
│   ├── csv/                             # 2qke_tmscore_values.csv, 5jyt_tmscore_values.csv, joint values CSV
│   └── plot/                            # 2qke_tmscore_1d_distribution.{png,pdf}, 5jyt_tmscore_1d_distribution.{png,pdf},
│                                        #   *_scatter_plddt, *_joint_plddt, combined_mfold_analysis.png
├── 02_voting/
│   ├── 2qke_tmscore/                    # voting_results.csv + sequence_voting_distribution.{png,pdf}
│   └── 5jyt_tmscore/                    # voting_results.csv + sequence_voting_distribution.{png,pdf}
├── 03_recompile/
│   ├── 2qke_tmscore/{prediction,control_prediction,plots}/
│   └── 5jyt_tmscore/{prediction,control_prediction,plots}/
├── 04_plots/
│   ├── 2qke_tmscore/bin_51_52/          # prediction_metrics.csv + control_prediction_metrics.csv
│   └── 5jyt_tmscore/bin_37_38_39_40_41/ # prediction_metrics.csv + control_prediction_metrics.csv
└── logs/af_claseq_pipeline.log
```

How to read them:

- **`01_m_fold_sampling/plot/` histograms & scatter** — the 1D TM-score histograms (one per reference) and the 2D scatter show the conformational landscape sampled across all groups. Because KaiB is bimodal, you should see populations near high `2qke_tmscore` (ground-fold-like) and, separately, near high `5jyt_tmscore` (fold-switch-like).
- **`02_voting/<metric>/voting_results.csv`** — bins are ranked by occupancy; the **high-count bins are the signal**. Sequences that repeatedly land in the high-`2qke_tmscore` bins are the ones encoding the **2QKE ground fold**; sequences dominating the high-`5jyt_tmscore` bins encode the **5JYT fold-switch state**. The bins selected in the YAML (`[51, 52]` for 2QKE, `[37–41]` for 5JYT) come straight from these results.
- **`04_plots/<metric>/bin_*/`** — the final per-bin `prediction_metrics.csv` (and its `control_prediction_metrics.csv`) are the payoff: the voted/purified sequences should cluster tightly at high TM-score to their target reference, well separated from the random control, confirming each sequence set reconstructs its intended fold.

## SLURM vs local

The YAML's `slurm:` block drives how the prediction stages dispatch jobs. The shipped values are placeholders you must fill in for your cluster:

```yaml
slurm:
  conda_env_path: "your_conda_env_path"   # path to your ColabFold conda env
  slurm_account:  "your_slurm_account"
  slurm_partition: "your_partition"
  slurm_gpus_per_task: 1
  slurm_cpus_per_task: 8
  slurm_time: "00:10:00"                   # demo jobs are tiny
  max_workers: 200                         # concurrent job cap
```

- **On a SLURM cluster:** set `conda_env_path`, `slurm_account`, and `slurm_partition` to real values and run the command above. `max_workers` caps how many ColabFold jobs are submitted concurrently.
- **On a local multi-GPU workstation (no SLURM):** use the interactive notebook `KaiB_mfold_demo.ipynb` in this folder, which runs the identical five-stage pipeline but distributes the per-group predictions directly across your local GPUs instead of submitting SLURM jobs. `conda_env_path` still needs to point at your ColabFold environment; the `slurm_*` fields are not used in that path.

## More

- Full workflow guide: [../../docs/m_fold_sampling_voting.md](../../docs/m_fold_sampling_voting.md) (and the **Workflow 3: M-Fold Sampling & Voting** section of the root [../../README.md](../../README.md)).
- Optional interactive walkthrough: open `KaiB_mfold_demo.ipynb` in this folder for the same pipeline with inline visualizations and the option to skip computation by loading the pre-computed results.
