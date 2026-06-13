# AF_ClaSeq: Leveraging Sequence Purification for Accurate Prediction of Multiple Conformational States with AlphaFold2

AF_ClaSeq is a bioinformatics toolkit for phylogenetically-guided protein structure prediction with AlphaFold/ColabFold. It purifies sequences from a multiple sequence alignment — via phylogenetic divide-and-conquer, leave-one-out validation, m-fold sampling, and occurrence voting — to coax AlphaFold2 into predicting multiple distinct conformational states of a protein.

## 📊 Reproduce the paper figures

**All source data and figures from the paper live in [`reproduce_results/`](reproduce_results/) — one folder per figure.** Each folder holds the figure image (`.png`), the CSV data behind every panel, and a short `README` mapping each file back to how it was generated (numerical data + figures only — no predicted PDB structures).

| Figure | Protein | Folder |
|:------:|---------|--------|
| 3 | Adenylate kinase | [`figure_3_adenylate_kinase`](reproduce_results/figure_3_adenylate_kinase) |
| 4 | ABL1 kinase | [`figure_4_abl1`](reproduce_results/figure_4_abl1) |
| 5 | GLP-1 receptor | [`figure_5_glp1r`](reproduce_results/figure_5_glp1r) |
| 6 | KaiB | [`figure_6_kaib`](reproduce_results/figure_6_kaib) |
| 7 | GB98 (designed protein) | [`figure_7_gb98`](reproduce_results/figure_7_gb98) |
| S1–S9 | AtaR · H-Ras · RfaH · XCL1 · Calmodulin · pyrophosphatase · PfMATE · MurJ · T1214 | [`supple_figure_1_atar` … `_9_t1214`](reproduce_results/) |

➡️ See **[`reproduce_results/README.md`](reproduce_results/README.md)** for the full per-figure index and provenance.

## ✨ Features

**Workflows:**
- **Divide-and-Conquer** — phylogenetically-guided clade splitting + structure prediction.
- **Leave-One-Out** — remove one sequence at a time to find which sequences drive prediction quality.
- **M-Fold Sampling & Voting** — random m-fold sampling, bin by structural metric, vote for the best bins, then re-predict.
- **Occurrence Voting** — count how often each sequence appears in high-quality structures and keep the top hits.
- **UMAP Voting** — VAE-embed predicted structures, joint UMAP with reference projection, Option-F binning, then re-predict.

**Key capabilities:**
- Distance-guided phylogenetic clade detection with coverage-based quality filtering.
- SLURM-based concurrent ColabFold job management for high-throughput prediction.
- Smart sequence grouping that avoids small remainder groups.
- Built-in structural analysis (RMSD / TM-score) and publication-quality plotting.

## 🛠️ Installation

**External prerequisites** (install and put on your `PATH` first):
- **FastTree** — phylogenetic tree construction · https://morgannprice.github.io/fasttree/
- **TM-align** — structure alignment / comparison · https://zhanggroup.org/TM-align/
- **ColabFold** (with GPU access) — AlphaFold structure prediction · https://github.com/sokrypton/ColabFold

**Install the package** (Python dependencies are declared in `pyproject.toml`):

```bash
git clone https://github.com/enmingxing244/AF_ClaSeq.git
cd AF_ClaSeq
pip install -e .
```

Verify with `python -c "import af_claseq; print('AF_ClaSeq installed!')"`, then point the config files at your ColabFold environment and FastTree binary.

## 🚀 Quick start — KaiB demo

A complete, runnable demo on KaiB (a fold-switching protein with two known conformational states) lives in [`example/KaiB_demo/`](example/KaiB_demo/). Run it **from inside that folder** (its config uses paths relative to the demo directory), on a SLURM cluster with GPU access:

```bash
cd example/KaiB_demo
python ../../scripts/run_m_fold_sampling_voting.py KaiB_m_fold_voting_demo.yaml
```

➡️ See **[`example/KaiB_demo/README.md`](example/KaiB_demo/README.md)** for the full walkthrough (including a no-SLURM local-GPU notebook and pre-computed results).

## 🧰 Command-line tools

| Command | Purpose |
|---------|---------|
| `python scripts/run_divide_and_conquer.py <config>` | Phylogenetic clade splitting + prediction |
| `python scripts/run_leave_one_out.py <config>` | Leave-one-out impact validation |
| `python scripts/run_m_fold_sampling_voting.py <config>` | M-fold sampling + voting |
| `python scripts/run_occurrence_voting.py <config>` | Occurrence-based sequence selection |
| `python scripts/run_vae_embedding.py <config>` | Train VAE on predicted structures |
| `python scripts/run_umap_voting.py <config>` | UMAP projection + Option-F voting |

Preprocessing & analysis helpers live in `scripts/utilities/`; an experimental MSA-column-shuffling tool is in `scripts/column_shuf/`.

## 📚 Workflows

| Workflow | Entry script | What it does | Guide |
|----------|--------------|--------------|-------|
| Divide-and-Conquer | `run_divide_and_conquer.py` | Splits the MSA into phylogenetic clades and predicts structures per clade | [docs/divide_and_conquer.md](docs/divide_and_conquer.md) |
| Leave-One-Out | `run_leave_one_out.py` | Removes one sequence at a time to rank each sequence's impact on prediction quality | [docs/leave_one_out.md](docs/leave_one_out.md) |
| M-Fold Sampling & Voting | `run_m_fold_sampling_voting.py` | Random sampling → bin by metric → vote for best bins → re-predict the winners | [docs/m_fold_sampling_voting.md](docs/m_fold_sampling_voting.md) |
| Occurrence Voting | `run_occurrence_voting.py` | Counts sequence occurrences in good structures and keeps the most frequent | [docs/occurrence_voting.md](docs/occurrence_voting.md) |
| UMAP Voting | `run_umap_voting.py` (+ `run_vae_embedding.py`) | VAE-embeds structures, projects onto reference UMAP, votes via Option F | [docs/umap_voting.md](docs/umap_voting.md) |

**Which workflow should I use?** Start with **Divide-and-Conquer** for a large MSA (>500 sequences) to get initial predictions. Use **Leave-One-Out** to discover which sequences matter, **Occurrence Voting** to narrow a pool down to an optimal subset, and **M-Fold Sampling** (or **UMAP Voting**) to map the conformational landscape of a small, curated set. The workflows chain naturally — each one's output feeds the next.

## 🗂️ Project structure

```
AF_ClaSeq/
├── src/af_claseq/            # installable package — one subpackage per workflow
│   ├── divide_and_conquer/   leave_one_out/   m_fold_sampling_voting/
│   ├── occurrence_voting/    umap_voting/ (+ vae/)
│   └── utils/                # shared helpers
├── scripts/                  # command-line entry points
│   ├── run_*.py              # the 6 workflow drivers
│   ├── utilities/            # preprocessing & analysis helpers
│   └── column_shuf/          # experimental MSA column-shuffling side-tool
├── docs/                     # detailed per-workflow guides
├── example/                  # KaiB demo + config templates
├── reproduce_results/        # paper source data + figures (one folder per figure)
└── tests/                    # unit tests
```

## ⚙️ Configuration

Workflows are driven by YAML config files (plus a JSON structure-analysis config that defines reference structures and metrics). Ready-to-edit templates for every workflow live in `example/config_examples/`. The full parameter reference is in **[docs/configuration.md](docs/configuration.md)**.

## 📖 Documentation

- [docs/divide_and_conquer.md](docs/divide_and_conquer.md) — Divide-and-Conquer workflow guide
- [docs/leave_one_out.md](docs/leave_one_out.md) — Leave-One-Out workflow guide
- [docs/m_fold_sampling_voting.md](docs/m_fold_sampling_voting.md) — M-Fold Sampling & Voting workflow guide
- [docs/occurrence_voting.md](docs/occurrence_voting.md) — Occurrence Voting workflow guide
- [docs/umap_voting.md](docs/umap_voting.md) — UMAP Voting workflow guide
- [docs/configuration.md](docs/configuration.md) — full configuration reference
- [docs/troubleshooting.md](docs/troubleshooting.md) — common issues and fixes
- [docs/background.md](docs/background.md) — scientific background and methods

## 📄 Citation

If you use AF_ClaSeq in your research, please cite:

```
@misc{xing2025leveragingsequencepurificationaccurate,
      title={Leveraging Sequence Purification for Accurate Prediction of Multiple Conformational States with AlphaFold2}, 
      author={Enming Xing and Junjie Zhang and Shen Wang and Xiaolin Cheng},
      year={2025},
      eprint={2503.00165},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2503.00165}, 
}
```
