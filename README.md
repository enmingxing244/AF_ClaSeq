# AF_ClaSeq

AF_ClaSeq is a research toolkit for identifying sequence subsets that steer AlphaFold2/ColabFold toward distinct protein conformations. It includes phylogenetic Divide-and-Conquer, Leave-One-Out analysis, M-fold sampling and voting, and occurrence voting.

## Choose an installation

| Goal | Install | ColabFold/GPU required? |
|------|---------|-------------------------|
| Reproduce plots or run analysis with the precomputed structures | Standard AF_ClaSeq installation | No |
| Generate structures from MSAs and reproduce the complete prediction workflow | Standard AF_ClaSeq installation plus a separate ColabFold environment | Yes |

Python 3.10 is the tested and recommended version.

### Standard installation: analysis without ColabFold

Use this option to analyze existing PDB files, run sequence voting, and reproduce figures from the shared precomputed-data archive.

```bash
git clone https://github.com/enmingxing244/AF_ClaSeq.git
cd AF_ClaSeq

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

python -c "import ete3; from af_claseq.m_fold_sampling_voting.config import load_pipeline_config; print(f'AF_ClaSeq installed successfully (ETE3 {ete3.__version__})')"
```

ETE3 3.1.3 or newer within the 3.x series is a core dependency used to parse Newick trees in Divide-and-Conquer. It is declared in `pyproject.toml` and installed automatically by `pip install -e .`; no separate ETE/ETE3 installation step is required.

Some workflows use an external executable in addition to the Python package:

- [TM-align](https://zhanggroup.org/TM-align/) is required when a structure-analysis JSON uses TM-score metrics, including the KaiB reproduction cases. Its executable must be available as `TMalign` on `PATH`.
- [FastTree](https://morgannprice.github.io/fasttree/) or `FastTreeMP` is required when constructing a new phylogenetic tree with Divide-and-Conquer. Set `input.fasttree_binary` in the workflow YAML to the executable's full path. FastTree is not required for M-fold analysis, voting, or plotting from precomputed structures.

If you use Conda, this tested alternative creates a single non-ColabFold environment containing Python, FastTree, and TM-align; the following `pip` command then installs AF_ClaSeq and its core Python dependencies:

```bash
conda create --name af-claseq --override-channels \
  --channel conda-forge --channel bioconda \
  python=3.10 pip fasttree tmalign
conda activate af-claseq
python -m pip install -e .
```

### Prediction setup: structure generation with ColabFold

First install AF_ClaSeq with either standard method above. Install ColabFold in a **separate environment** by following the current [official ColabFold installation instructions](https://github.com/sokrypton/ColabFold), then activate that environment and verify that the prediction command is available:

```bash
command -v colabfold_batch
```

The separate environments avoid dependency conflicts between this publication code and current ColabFold releases. Point the workflow YAML's ColabFold environment setting (for example, `slurm.conda_env_path`) to that environment. AF_ClaSeq currently submits prediction stages through SLURM, so full prediction runs also require a GPU cluster and site-appropriate SLURM account, partition, environment, time-limit, and module settings. The standard installation above is sufficient for the precomputed-data reproduction workflow.

## Reproduce the manuscript results

There are two levels of released results:

1. [`reproduce_results/`](reproduce_results/) contains the manuscript figures and their numerical source data. See [`reproduce_results/README.md`](reproduce_results/README.md) for the figure-to-file index.
2. The larger `data_af_claseq` archive contains the precomputed PDB structures, portable YAML/JSON configurations, CSV files, and plot inputs needed to rerun the analysis pipelines without ColabFold.

> **Precomputed-data archive:** OneDrive download link to be added after upload (`DATA_ARCHIVE_LINK_TODO`).

Download and extract the archive so that it sits directly below the repository root:

```text
AF_ClaSeq/
├── scripts/
├── src/
└── data_af_claseq/
    ├── ABL1/
    ├── AdK/
    ├── GB98/
    ├── GLP1R/
    ├── KaiB/
    └── README.md
```

Then run all commands from the repository root:

```bash
cd /path/to/AF_ClaSeq
source .venv/bin/activate
test -f scripts/run_m_fold_sampling_voting.py
test -d data_af_claseq
```

The downloaded archive includes `data_af_claseq/README.md`, which provides the exact copy-and-paste command for every manuscript case and explains the safe plot-only stages, optional CPU voting stage, and full ColabFold stages.

## Command-line workflows

| Workflow | Command | Detailed guide |
|----------|---------|----------------|
| Divide-and-Conquer | `python scripts/run_divide_and_conquer.py --config <config.yaml>` | [Guide](docs/divide_and_conquer.md) |
| Leave-One-Out | `python scripts/run_leave_one_out.py <config.yaml>` | [Guide](docs/leave_one_out.md) |
| M-fold sampling and voting | `python scripts/run_m_fold_sampling_voting.py <config.yaml>` | [Guide](docs/m_fold_sampling_voting.md) |
| Occurrence voting | `python scripts/run_occurrence_voting.py <config.yaml>` | [Guide](docs/occurrence_voting.md) |

Configuration templates are available in [`example/config_examples/`](example/config_examples/). The complete parameter reference is in [`docs/configuration.md`](docs/configuration.md), and common errors are covered in [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Repository layout

```text
AF_ClaSeq/
├── src/af_claseq/          # Python package
├── scripts/                # Command-line entry points and utilities
├── docs/                   # Workflow and configuration guides
├── example/config_examples # Portable configuration templates
├── reproduce_results/      # Manuscript figures and numerical source data
└── tests/                  # Automated tests
```

## Citation

If you use AF_ClaSeq, please cite:

```bibtex
@misc{xing2025leveragingsequencepurificationaccurate,
  title         = {Leveraging Sequence Purification for Accurate Prediction of Multiple Conformational States with AlphaFold2},
  author        = {Enming Xing and Junjie Zhang and Shen Wang and Xiaolin Cheng},
  year          = {2025},
  eprint        = {2503.00165},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.BM},
  url           = {https://arxiv.org/abs/2503.00165}
}
```

AF_ClaSeq is released under the [MIT License](LICENSE).
