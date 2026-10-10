"""OpenFold prediction-engine helpers for AF_ClaSeq.

This module contains the glue needed to run OpenFold as an alternative to ColabFold:
converting a ColabFold-style ``task_dir`` (a folder of ``*.a3m`` MSAs) into OpenFold's
batch input layout, building the ``run_pretrained_openfold.py`` command + its SLURM
environment prelude, and collecting the resulting PDBs back under ColabFold's naming
convention so the rest of the pipeline is untouched.

Design notes
------------
* **No heavy imports.** Everything here is pure file I/O / string building (stdlib only),
  so the module loads in the *driver* conda env (the one that has ``af_claseq``). The a3m
  conversion and PDB collection run driver-side; only the inference itself runs inside the
  SLURM job, in the isolated ``openfold2`` env (which deliberately does **not** contain
  ``af_claseq``). This keeps the two dependency stacks separated.
* **DeepSpeed + BF16 by default.** Benchmarked 3.6x faster than ColabFold on real ClaSeq
  chunks at equivalent quality. See ``af-claseq-openfold/OPENFOLD_INTEGRATION.md``.
* Paths are config-driven: the constants below are defaults that callers can override.
"""

import os
import glob
import shutil
from typing import Dict, List, Optional, Tuple

from af_claseq.utils.logging_utils import get_logger

logger = get_logger("openfold_utils")


# --- Default paths (overridable via config / SlurmJobSubmitter kwargs) ---------------
DEFAULT_OPENFOLD_DIR = "/fs/ess/PAA0203/xing244/af-claseq-openfold/packages/openfold"
DEFAULT_OPENFOLD_CONDA_ENV = "/fs/ess/PAA0203/xing244/.conda/envs/openfold2"

# --- Verified OpenFold configurations ------------------------------------------------
# Maps a config name -> the *extra* CLI flags for run_pretrained_openfold.py. The model
# preset (``--config_preset``) is added separately so the engine config and the model
# choice stay orthogonal. ``{flash_config}`` is substituted with a real JSON file path
# (written into the work dir on demand) for the FlashAttention variants.
OPENFOLD_CONFIGS: Dict[str, List[str]] = {
    "default": ["--precision", "fp32"],
    "deepspeed_bf16": ["--use_deepspeed_evoformer_attention", "--precision", "bf16"],
    "deepspeed": ["--use_deepspeed_evoformer_attention"],
    "bf16": ["--precision", "bf16"],
    "flash_attention": ["--experiment_config_json", "{flash_config}"],
    "bf16_flash": ["--precision", "bf16", "--experiment_config_json", "{flash_config}"],
}

DEFAULT_OPENFOLD_CONFIG = "deepspeed_bf16"
DEFAULT_OPENFOLD_MODEL = "model_3_ptm"

# Layout / sentinel constants
OPENFOLD_WORK_DIRNAME = "_openfold_work"
FLASH_CONFIG_CONTENT = '{"globals.use_flash": true}'
# Minimal valid mmCIF data block. model_3_ptm is template-free, but the CLI parser still
# requires the template dir to contain a file (real dir on disk is 11 bytes: "data_dummy\n").
DUMMY_CIF_CONTENT = "data_dummy\n"


def openfold_work_dir(task_dir: str) -> str:
    """Return the per-task OpenFold scratch dir (lives inside ``task_dir``)."""
    return os.path.join(task_dir, OPENFOLD_WORK_DIRNAME)


def openfold_output_dir(task_dir: str) -> str:
    """Return the OpenFold ``--output_dir`` for a task (inside the work dir)."""
    return os.path.join(openfold_work_dir(task_dir), "output")


def parse_query_from_a3m(a3m_path: str) -> str:
    """Extract the ungapped query sequence (the first record) from an a3m file.

    Skips comment (``#``) lines, takes the first ``>``-headed record, and strips alignment
    gap characters (``-`` and ``.``). Mirrors the reference conversion in
    ``af-claseq-openfold/scripts/05_chunk_msa.py``.
    """
    seq_parts: List[str] = []
    in_query = False
    with open(a3m_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if in_query:  # second header reached -> query record finished
                    break
                in_query = True
                continue
            if in_query:
                seq_parts.append(line)
    return "".join(seq_parts).replace("-", "").replace(".", "")


def prepare_openfold_input(
    task_dir: str,
    work_dir: str,
    openfold_config: str = DEFAULT_OPENFOLD_CONFIG,
) -> Tuple[str, str, str]:
    """Convert a ColabFold-style ``task_dir`` into OpenFold's batch input layout.

    Creates, inside ``work_dir``:
      ``fasta/``          one ``<tag>.fasta`` per a3m (ungapped query only)
      ``alignments/<tag>/`` ``bfd_uniclust_hits.a3m`` (verbatim copy of the a3m) plus
                          ``uniref90_hits.sto`` / ``mgnify_hits.sto`` query stubs
      ``template_mmcif/`` a dummy template dir (required by the CLI, unused by model_3_ptm)

    The conversion is a clean rebuild (idempotent across resubmissions). Returns
    ``(fasta_dir, alignment_dir, template_dir)``.
    """
    fasta_dir = os.path.join(work_dir, "fasta")
    alignment_dir = os.path.join(work_dir, "alignments")
    template_dir = os.path.join(work_dir, "template_mmcif")
    for d in (fasta_dir, alignment_dir, template_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    a3m_files = sorted(f for f in os.listdir(task_dir) if f.endswith(".a3m"))
    if not a3m_files:
        raise FileNotFoundError(f"No .a3m files found in {task_dir}")

    for a3m_name in a3m_files:
        tag = os.path.splitext(a3m_name)[0]
        a3m_path = os.path.join(task_dir, a3m_name)
        query = parse_query_from_a3m(a3m_path)
        if not query:
            raise ValueError(f"Could not parse a query sequence from {a3m_path}")

        with open(os.path.join(fasta_dir, f"{tag}.fasta"), "w") as fh:
            fh.write(f">{tag}\n{query}\n")

        tag_align_dir = os.path.join(alignment_dir, tag)
        os.makedirs(tag_align_dir, exist_ok=True)
        # Full a3m content -> the actual MSA OpenFold reads.
        shutil.copyfile(a3m_path, os.path.join(tag_align_dir, "bfd_uniclust_hits.a3m"))
        # Stockholm stubs MUST contain the query sequence (an empty .sto crashes OpenFold).
        for stub in ("uniref90_hits.sto", "mgnify_hits.sto"):
            with open(os.path.join(tag_align_dir, stub), "w") as fh:
                fh.write(f"# STOCKHOLM 1.0\n{tag} {query}\n//\n")

    with open(os.path.join(template_dir, "dummy.cif"), "w") as fh:
        fh.write(DUMMY_CIF_CONTENT)

    # FlashAttention variants need a real JSON file path.
    if "flash" in openfold_config:
        _write_flash_config(work_dir)

    return fasta_dir, alignment_dir, template_dir


def _write_flash_config(work_dir: str) -> str:
    """Write the FlashAttention experiment-config JSON and return its path."""
    path = os.path.join(work_dir, "flash_config.json")
    os.makedirs(work_dir, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(FLASH_CONFIG_CONTENT)
    return path


def build_openfold_command(
    fasta_dir: str,
    template_dir: str,
    output_dir: str,
    alignment_dir: str,
    openfold_dir: str = DEFAULT_OPENFOLD_DIR,
    openfold_config: str = DEFAULT_OPENFOLD_CONFIG,
    openfold_model: str = DEFAULT_OPENFOLD_MODEL,
    save_outputs: bool = False,
    flash_config_path: Optional[str] = None,
) -> str:
    """Build the ``run_pretrained_openfold.py`` invocation (the part after the env setup).

    ``save_outputs`` is off by default: the per-chunk output pickle is ~260 MB and the
    pipeline only needs the PDB structure.
    """
    if openfold_config not in OPENFOLD_CONFIGS:
        raise ValueError(
            f"Unknown openfold_config '{openfold_config}'. "
            f"Valid options: {sorted(OPENFOLD_CONFIGS)}"
        )
    if flash_config_path is None and "flash" in openfold_config:
        flash_config_path = os.path.join(os.path.dirname(output_dir), "flash_config.json")

    config_args = [
        arg.replace("{flash_config}", flash_config_path or "")
        for arg in OPENFOLD_CONFIGS[openfold_config]
    ]

    parts = [
        "python",
        os.path.join(openfold_dir, "run_pretrained_openfold.py"),
        fasta_dir,
        template_dir,
        "--output_dir", output_dir,
        "--use_precomputed_alignments", alignment_dir,
        "--config_preset", openfold_model,
        "--model_device", "cuda:0",
        "--skip_relaxation",
    ]
    if save_outputs:
        parts.append("--save_outputs")
    parts.extend(config_args)
    return " ".join(parts)


def build_openfold_env_setup(
    openfold_conda_env: str = DEFAULT_OPENFOLD_CONDA_ENV,
    openfold_dir: str = DEFAULT_OPENFOLD_DIR,
) -> str:
    """Build the bash environment prelude for the OpenFold SLURM ``--wrap``.

    Mirrors the proven benchmark setup, adapted to AF_ClaSeq's module conventions:
    * ``set -eo pipefail`` only -- never ``set -u`` (conda's gcc activate script has an
      unbound ``SYS_SYSROOT`` variable that would abort the job).
    * no ``conda init`` (it rewrites ~/.bashrc non-atomically and has truncated this
      user's ~/.bashrc under concurrent OSC jobs).
    * ``CUTLASS_PATH`` is required by DeepSpeed's DS4Sci attention kernel.
    * ``LD_LIBRARY_PATH`` must include ``$CONDA_PREFIX/lib`` for shared libraries.
    * ``run_pretrained_openfold.py`` must be invoked from its source dir (it imports
      local modules), hence the trailing ``cd``.
    """
    cue_ops_lib = os.path.join(
        openfold_conda_env, "lib", "python3.10", "site-packages",
        "cuequivariance_ops", "lib",
    )
    cutlass_path = os.path.join(openfold_dir, "cutlass")
    return (
        "set -eo pipefail && "
        "module reset && module load cuda/12.4.1 miniconda3/24.1.2-py310 && "
        f"conda activate {openfold_conda_env} && "
        f"export CUTLASS_PATH={cutlass_path} && "
        "export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH && "
        f"export LD_LIBRARY_PATH={cue_ops_lib}:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && "
        f"cd {openfold_dir}"
    )


def build_openfold_wrap(
    task_dir: str,
    openfold_dir: str = DEFAULT_OPENFOLD_DIR,
    openfold_conda_env: str = DEFAULT_OPENFOLD_CONDA_ENV,
    openfold_config: str = DEFAULT_OPENFOLD_CONFIG,
    openfold_model: str = DEFAULT_OPENFOLD_MODEL,
    save_outputs: bool = False,
) -> str:
    """Prepare the OpenFold input for ``task_dir`` and return the full sbatch ``--wrap``.

    This runs the a3m->OpenFold conversion *driver-side* (no torch needed) and returns the
    ``"<env setup> && <python run_pretrained_openfold.py ...>"`` string to hand to sbatch.
    """
    work_dir = openfold_work_dir(task_dir)
    fasta_dir, alignment_dir, template_dir = prepare_openfold_input(
        task_dir, work_dir, openfold_config=openfold_config
    )
    output_dir = openfold_output_dir(task_dir)
    os.makedirs(output_dir, exist_ok=True)
    flash_config_path = (
        os.path.join(work_dir, "flash_config.json") if "flash" in openfold_config else None
    )
    command = build_openfold_command(
        fasta_dir=fasta_dir,
        template_dir=template_dir,
        output_dir=output_dir,
        alignment_dir=alignment_dir,
        openfold_dir=openfold_dir,
        openfold_config=openfold_config,
        openfold_model=openfold_model,
        save_outputs=save_outputs,
        flash_config_path=flash_config_path,
    )
    env_setup = build_openfold_env_setup(openfold_conda_env, openfold_dir)
    return f"{env_setup} && {command}"


def collect_openfold_output(
    task_dir: str,
    openfold_model: str = DEFAULT_OPENFOLD_MODEL,
    seed: int = 0,
) -> int:
    """Copy OpenFold PDBs back into ``task_dir`` under ColabFold's naming convention.

    OpenFold writes ``<output>/predictions/<tag>_<model>_unrelaxed.pdb``. Each is copied to
    ``<task_dir>/<tag>_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_<seed>.pdb`` so that
    downstream code (which globs ``*.pdb`` and splits filenames on ``_unrelaxed``) finds the
    structures unchanged. The ``seed`` suffix mirrors the submitter's configured random seed
    (stages such as sequence_voting match on the exact seed). After copying, the entire work
    dir is removed so the recursive ``*.pdb`` scans never see the raw (differently-named)
    OpenFold outputs and double-count them.

    Returns the number of PDBs collected.
    """
    work_dir = openfold_work_dir(task_dir)
    output_dir = openfold_output_dir(task_dir)
    suffix = f"_{openfold_model}_unrelaxed.pdb"
    pdb_paths = sorted(
        glob.glob(os.path.join(output_dir, "**", f"*{suffix}"), recursive=True)
    )

    collected = 0
    for pdb_path in pdb_paths:
        name = os.path.basename(pdb_path)
        tag = name[: -len(suffix)] if name.endswith(suffix) else name.split("_unrelaxed")[0]
        dest = os.path.join(
            task_dir,
            f"{tag}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_{seed:03d}.pdb",
        )
        shutil.copyfile(pdb_path, dest)
        collected += 1

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    return collected


def write_done_log(task_dir: str) -> None:
    """Write the synthetic completion sentinel (ColabFold writes ``log.txt`` with 'Done';
    OpenFold does not, so the pipeline's completion check needs us to emit it)."""
    with open(os.path.join(task_dir, "log.txt"), "w") as fh:
        fh.write("OpenFold prediction complete.\nDone\n")
