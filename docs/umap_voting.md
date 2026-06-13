# UMAP Voting Workflow

VAE-embed predicted structures, jointly UMAP-project them with reference structures, bin on the UMAP grid using Option F (top-K most frequent sequences per reference bin), re-fold with ColabFold, and evaluate the predictions with an RMSD scatter.

[← README](../README.md) · [Workflows](divide_and_conquer.md) · [LOO](leave_one_out.md) · [M-Fold](m_fold_sampling_voting.md) · [Occurrence](occurrence_voting.md) · [UMAP](umap_voting.md) · [Config](configuration.md) · [Troubleshooting](troubleshooting.md)

**When to use**: You have a pool of predicted structures (e.g., from M-Fold sampling) and want to find which sequences map near each reference conformation in a learned structural embedding, then re-predict those sequences in detail.

**What it does**: Trains a variational autoencoder (VAE) on per-structure coordinate features, encodes every structure into a latent embedding, projects those embeddings together with reference structures into a shared 2D UMAP, then bins the UMAP grid. For each reference bin it keeps the **top-K most frequent sequences** (Option F voting), re-predicts them with ColabFold, and plots RMSD-vs-reference scatters to assess how well each bin reproduces its target conformation.

This is a **two-stage** pipeline with two entry points:

```bash
# Stage A: train the VAE and write the structure embeddings
python scripts/run_vae_embedding.py vae_embedding.yaml

# Stage B: UMAP-project, vote, predict, and scatter
python scripts/run_umap_voting.py umap_voting.yaml
```

## Before You Start

**You need**:
1. ✅ A CSV of predicted structures to embed (`inputs.structures_csv`) — the structure pool to analyze
2. ✅ A references CSV describing the reference conformations (`inputs.references_csv`)
3. ✅ A structure analysis config JSON (`structure_analysis.config_json`) defining the metrics/references
4. ✅ A query `.a3m` for the sequence being modeled (`inputs.query_a3m`)
5. ✅ ColabFold environment + SLURM access (for the predict stage)
6. ✅ A GPU is recommended for VAE training (`general.device: "cuda"`)

The two stages are connected by the VAE embedding file: stage A writes `embedding.npz` (under the VAE `base_dir`, named by `output.embedding_filename`), and stage B reads it via `inputs.embedding_npz`.

## Stage A: VAE Embedding

### Configure

Copy and edit the example:

```bash
cp example/config_examples/vae_embedding.yaml my_vae_config.yaml
```

```yaml
general:
  protein_name: "KRAS_HUMAN"
  base_dir: "/path/to/output"
  random_seed: 42
  device: "cuda"                    # "cpu" | "cuda"

inputs:
  structures_csv: "/path/to/structures.csv"
  references_csv: "/path/to/references.csv"

structure_analysis:
  config_json: "/path/to/structure_analysis_config.json"
  coord_target: "local"             # "local" | "global" — which coordinate set to embed

coord_extraction:
  alignment_ref_pdb: null           # optional: PDB to Kabsch-align all structures to
  alignment_ref_chain: "A"
  target_chain: "A"                 # chain to extract from sampling PDBs

vae:
  model:
    latent_dim: 16
    hidden_channels: [32, 64, 128]
    use_residual: true
  training:
    epochs: 1000
    batch_size: 256
    learning_rate: 1.0e-3
    kl_weight: 0.1
    val_split: 0.1
    save_best_only: true
    early_stopping_patience: 50

output:
  embedding_filename: "embedding.npz"
```

**Key fields**:
- `structure_analysis.coord_target` selects which coordinate set (`"local"` or `"global"`) the VAE trains on.
- `coord_extraction.alignment_ref_pdb` (optional) Kabsch-aligns every structure to a common reference before feature extraction; `target_chain` is the chain pulled from each sampling PDB.
- `vae.model` sets the latent dimensionality and the convolutional encoder channels; `vae.training` controls epochs, batch size, KL weighting, validation split, and early stopping.

### Run

```bash
# Validate the config without training
python scripts/run_vae_embedding.py --validate-only my_vae_config.yaml

# Train and encode (optionally override the device)
python scripts/run_vae_embedding.py my_vae_config.yaml
python scripts/run_vae_embedding.py --device cuda my_vae_config.yaml
```

**Output**: the embedding file (`embedding.npz` by default) containing the latent vector for every structure. Point `inputs.embedding_npz` in the stage-B config at this file.

## Stage B: UMAP Voting

### Configure

Copy and edit the example:

```bash
cp example/config_examples/umap_voting.yaml my_umap_config.yaml
```

```yaml
general:
  protein_name: "KRAS_HUMAN"
  base_dir: "/path/to/output"
  random_seed: 42

inputs:
  embedding_npz: "/path/to/vae/embedding.npz"   # from Stage A
  references_csv: "/path/to/references.csv"
  query_a3m: "/path/to/query.a3m"
  rmsd_vs_refs_csv: null   # optional, for UMAP coloring

umap:
  n_neighbors: 30
  min_dist: 0.1
  n_components: 2
  metric: "euclidean"
  umap1_range: [1.0, 11.0]   # null = auto, then pinned in grid.json
  umap2_range: [1.0, 9.0]

binning:
  bin_size: 1.0
  top_k: 16                  # Option F: top-K most frequent sequences per ref bin
  min_records_per_bin: 50

structure_prediction:
  num_models: 5
  num_seeds: 8
  num_recycle: 3
  prediction_mode: "monomer"   # "monomer" | "homodimer"
  rank: "plddt"
  random_seed: 0

slurm:
  conda_env_path: "/fs/ess/PAA0203/xing244/.conda/envs/colabfold"
  account: "PAA0203"
  partition: "nextgen"
  time: "00:30:00"
  gpus_per_task: 1
  cpus_per_task: 4
  check_interval: 60

structure_analysis:
  config_json: "/path/to/structure_analysis_config.json"
  metrics: ["local", "global"]

plotting:
  formats: ["png", "pdf"]
  metric_ranges:
    local:  { min: 0, max: 6, ticks: [0, 1, 2, 3, 4, 5, 6] }
    global: { min: 0, max: 4, ticks: [0, 1, 2, 3, 4] }
  colors:
    ref1: "#C73E3A"
    ref2: "#2E7AB8"
  panels_per_row: 2
```

**Key fields**:
- `umap.umap1_range` / `umap2_range` fix the projection extent; leave them `null` to auto-fit (the chosen range is then pinned in `grid.json`).
- `binning.bin_size` sets the UMAP grid resolution; `binning.top_k` is the Option F parameter — the number of most-frequent sequences kept per reference bin; `min_records_per_bin` skips sparsely populated bins.
- `structure_prediction` controls the ColabFold re-prediction (models, seeds, recycles, monomer vs homodimer).
- `structure_analysis.metrics` lists the RMSD metrics to plot; `plotting` sets output formats, per-metric axis ranges, reference colors, and panel layout.

### Run

The pipeline has four stages, run in order:

```
project → vote → predict → scatter
```

| Stage | What it does | Output |
|-------|--------------|--------|
| `project` | Joint UMAP of structure embeddings + references; assign UMAP-grid bins | `base_dir/umap/` (`umap_coords.csv`, `grid.json`) |
| `vote` | Option F: keep top-K most frequent sequences per reference bin, write per-bin A3Ms | `base_dir/voting/` (`voting_summary.csv`, `a3ms/`) |
| `predict` | Submit ColabFold jobs for the voted A3Ms | `base_dir/predictions/` (`job_manifest.csv`) |
| `scatter` | RMSD-vs-reference scatter for the re-predictions | `base_dir/scatter/` (`per_pred.csv`, figures) |

```bash
# Validate the config without running
python scripts/run_umap_voting.py --validate-only my_umap_config.yaml

# Run the full pipeline
python scripts/run_umap_voting.py my_umap_config.yaml

# Dry run: project + vote only (skips predict + scatter)
python scripts/run_umap_voting.py --dry-run my_umap_config.yaml

# Run a single stage, or a sub-range
python scripts/run_umap_voting.py --start-from vote --stop-after vote my_umap_config.yaml
python scripts/run_umap_voting.py --start-from predict my_umap_config.yaml

# Force the UMAP to refit even if a persisted model exists
python scripts/run_umap_voting.py --refit-umap my_umap_config.yaml
```

**Resume behavior**: stages whose outputs already exist are skipped automatically when resuming, so you can re-run safely. Use `--start-from` / `--stop-after` to control the range explicitly, and `--refit-umap` to discard a cached UMAP model and recompute the projection.

## Outputs

After the full pipeline completes, results live under `base_dir`:

```bash
# UMAP projection + grid
ls base_dir/umap/            # umap_coords.csv, grid.json

# Option F voting results and per-bin A3Ms
head base_dir/voting/voting_summary.csv
ls base_dir/voting/a3ms/

# ColabFold re-prediction manifest
head base_dir/predictions/job_manifest.csv

# RMSD scatter data + figures
head base_dir/scatter/per_pred.csv
ls base_dir/scatter/
```

**Interpreting the scatter**: each point is one re-predicted structure plotted against its RMSD to the reference conformations. Bins whose predictions cluster at low RMSD to a given reference reproduce that conformation well; the colors and per-metric axis ranges come from the `plotting` section of the config.

---
[← Back to README](../README.md)
