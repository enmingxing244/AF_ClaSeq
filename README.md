# AF_ClaSeq: Leveraging Sequence Purification for Accurate Prediction of Multiple Conformational States with AlphaFold2

AlphaFold2 (AF2) has transformed protein structure prediction by harnessing co-evolutionary constraints embedded in multiple sequence alignments (MSAs). MSAs not only encode static structural information, but also contain evolutionary signatures linked to protein dynamics and conformational heterogeneity, which are central to biological function. However, the subtle co-evolutionary signals that bias proteins toward distinct conformational states are often obscured by noise within MSA data and remain challenging to decipher. Here, we introduce AF-ClaSeq, a systematic framework that enriches and purifies state-specific evolutionary signals through iterative sequence classification and selection. By identifying sequence subsets that preferentially encode distinct conformational states, AF-ClaSeq enables robust predictions of alternative conformations across diverse protein systems. Our findings indicate that the accuracy of alternative-state prediction depends more on the purity of the evolutionary signal within an MSA than on sequence depth. Notably, purified sequences encoding specific structural states are distributed across phylogenetic clades and superfamilies, rather than being confined to individual lineages. By extending AF2 beyond single-state structure prediction, AF-ClaSeq provides a promising approach for exploring protein structural plasticity, establishing a foundation for future studies of the sequence determinants underlying conformational switching and allosteric regulation.

## Choose an installation

| Goal | Install | ColabFold/GPU required? |
|------|---------|-------------------------|
| Inspect the numerical source data | Excel-compatible spreadsheet application | No |
| Reproduce plots or run analysis with the precomputed structures | Standard AF_ClaSeq installation | No |
| Generate structures from MSAs and reproduce the complete prediction workflow | Standard AF_ClaSeq installation plus a separate ColabFold environment | Yes |

> **Precomputed-data archive:** [Download the complete `data_af_claseq` raw-data folder from OneDrive](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/xing_244_osu_edu/IgCct8g4giIaSYkqAZVbiahoAeQ4ZepXd2br7HnP-nFSeCY?e=8ysLeU).

Use Python 3.10 for the AF_ClaSeq analysis environment. The commands below use a Bash-compatible shell on Linux. Choose either the virtual environment or the Conda environment below.

### Standard installation: analysis without ColabFold

Use this option to analyze existing PDB files, run sequence voting, and reproduce figures from the shared precomputed-data archive.

```bash
git clone https://github.com/enmingxing244/AF_ClaSeq.git
cd AF_ClaSeq

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

python -c "import ete3; from af_claseq.m_fold_sampling_voting.config import load_pipeline_config; print(f'AF_ClaSeq installed successfully (ETE3 {ete3.__version__})')"
```

ETE3 3.1.3 or newer within the 3.x series is a core dependency used to parse Newick trees in Divide-and-Conquer. It is declared in `pyproject.toml` and installed automatically by `pip install -e .`; no separate ETE/ETE3 installation step is required.

### Install the third-party executables

FastTree and TM-align are independent third-party programs. As an alternative to the virtual environment above, run the following from the cloned repository root to create a Conda environment containing Python, both executables, and AF_ClaSeq. If `.venv` is active, run `deactivate` first:

```bash
conda create --name af-claseq --override-channels \
  --channel conda-forge --channel bioconda \
  --strict-channel-priority \
  python=3.10 pip fasttree tmalign
conda activate af-claseq
python -m pip install -e .

command -v FastTreeMP || command -v FastTree
command -v TMalign
```

- [TM-align](https://zhanggroup.org/TM-align/) is required when a structure-analysis JSON uses TM-score metrics, including the KaiB reproduction cases. AF_ClaSeq expects the executable name `TMalign` on `PATH`.
- [FastTree](https://morgannprice.github.io/fasttree/) or its multithreaded executable `FastTreeMP` is required only when Divide-and-Conquer constructs a new phylogenetic tree. Set `input.fasttree_binary` in the workflow YAML to the executable's full path. FastTree is not required for M-fold analysis, voting, or plotting from precomputed structures. The Divide-and-Conquer tree-building stage submits a CPU SLURM job, so that workflow also requires SLURM access.

If Conda is unavailable, use the authors' official download/build instructions linked above. FastTree provides Linux executables and `FastTree.c`; TM-align provides `TMalign.cpp`. Their documented Linux source-build commands are:

```bash
gcc -O3 -fopenmp-simd -funsafe-math-optimizations -march=native \
  -o FastTree FastTree.c -lm
g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp
```

Omit `-static` on systems that do not support static linking. Put the resulting `FastTree`/`FastTreeMP` and `TMalign` executables on `PATH`, or give the full FastTree path in the YAML. Do not copy these third-party binaries into the AF_ClaSeq repository.

### Prediction setup: structure generation with ColabFold

First install AF_ClaSeq with either standard method above. The manuscript predictions used **ColabFold 1.5.5** on an NVIDIA A100 GPU (40 GB). For reproducing those predictions, use that ColabFold version in a **separate environment** with compatible CUDA/JAX dependencies. The [official ColabFold installation instructions](https://github.com/sokrypton/ColabFold) describe the current release, which may differ from the manuscript version. Activate the prediction environment and verify that the command is available:

```bash
command -v colabfold_batch
```

Point the workflow YAML's ColabFold environment setting (for example, `slurm.conda_env_path`) to the prediction environment. Run AF_ClaSeq commands from the analysis environment. Full prediction runs require a GPU cluster and site-appropriate SLURM account, partition and time-limit settings. The shared [SLURM helper](src/af_claseq/utils/slurm_utils.py) also hardcodes `cuda/12.4.1` and `miniconda3/24.1.2-py310` module loads in `env_setup`; adapt that setup to your cluster before submitting predictions. Changing the YAML alone does not change these module loads. The plot-only reproduction workflow below does not submit SLURM jobs.

### Optional VAE and UMAP workflows

Install the additional dependencies in the AF_ClaSeq analysis environment:

```bash
python -m pip install -e ".[umap-voting]"
```

The [VAE/UMAP guide](docs/umap_voting.md) describes embedding training and UMAP-based sequence voting. VAE training can use CPU or GPU; the subsequent ColabFold prediction stage requires the separate prediction environment.

## Reproduce the manuscript results

The numerical source data and precomputed analysis inputs are available separately:

1. [`source_data/`](source_data/) contains the final numerical source data for the manuscript figures as Excel workbooks, with one workbook per figure and figure numbers in the filenames.
2. The larger `data_af_claseq` archive contains precomputed PDB structures, YAML/JSON configurations, CSV files and plot inputs for the five main-figure case studies: AdK, ABL1, GLP1R, KaiB and GB98. Its plot-only configurations run without ColabFold.

Access to the OneDrive share may require Microsoft sign-in and permission to the shared folder. The Excel workbooks in `source_data/` are included directly in this repository.

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

Return to the repository root and activate the analysis environment you created: `source .venv/bin/activate` for the virtual environment, or `conda activate af-claseq` for Conda. Check the data layout:

```bash
cd /path/to/AF_ClaSeq
test -f scripts/run_m_fold_sampling_voting.py
test -d data_af_claseq
```

For example, reproduce the AdK analysis with:

```bash
python scripts/run_m_fold_sampling_voting.py data_af_claseq/AdK/KAD_m_fold_sampling_voting.yaml
```

The distributed AdK configuration enables only `01_M_FOLD_SAMPLING_PLOT` and `04_PURE_SEQ_PLOT_RUN`, which analyze the saved predictions and write CSV files, plots and logs under `general.base_dir`. Run in an extracted working copy to preserve the downloaded files.

The archive includes `data_af_claseq/README.md` with commands for all five main-figure case studies, including their separate sampling rounds. It explains the plot-only stages, optional CPU voting stage and full prediction stages. The first GB98 round enables only `01_M_FOLD_SAMPLING_PLOT` because it has no recompiled predictions.

## Command-line workflows

| Workflow | Command | Detailed guide |
|----------|---------|----------------|
| Divide-and-Conquer | `python scripts/run_divide_and_conquer.py --config <config.yaml>` | [Guide](docs/divide_and_conquer.md) |
| Leave-One-Out | `python scripts/run_leave_one_out.py <config.yaml>` | [Guide](docs/leave_one_out.md) |
| M-fold sampling and voting | `python scripts/run_m_fold_sampling_voting.py <config.yaml>` | [Guide](docs/m_fold_sampling_voting.md) |
| Occurrence voting | `python scripts/run_occurrence_voting.py <config.yaml>` | [Guide](docs/occurrence_voting.md) |
| VAE embedding | `python scripts/run_vae_embedding.py <config.yaml>` | [Guide](docs/umap_voting.md#stage-a-vae-embedding) |
| UMAP voting | `python scripts/run_umap_voting.py <config.yaml>` | [Guide](docs/umap_voting.md#stage-b-umap-voting) |

Templates for the first four workflows are available in [`example/config_examples/`](example/config_examples/); VAE and UMAP YAML examples are provided in their guide. The complete parameter reference is in [`docs/configuration.md`](docs/configuration.md), and common errors are covered in [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Repository layout

```text
AF_ClaSeq/
├── src/af_claseq/           # Python package
├── scripts/                # Command-line entry points and utilities
├── docs/                   # Workflow and configuration guides
├── example/config_examples # Portable configuration templates
├── source_data/            # Final numerical source data Excel workbooks
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
