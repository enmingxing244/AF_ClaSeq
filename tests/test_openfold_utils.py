"""Unit tests for the OpenFold prediction-engine helpers (pure file-I/O + command building).

These run entirely driver-side -- no GPU, no SLURM, no torch -- exercising the a3m ->
OpenFold conversion, the command/env construction, and the PDB collection/renaming.
"""
import os

import pytest

from af_claseq.utils import openfold_utils as ofu


# --------------------------------------------------------------------------- helpers
A3M_GROUP_1 = (
    "#27\t1\n"
    ">101\n"
    "ACDEF---GHIKLMNPQR\n"
    ">hitA\n"
    "ACDEFxxxGHIKLMNPQR\n"
    ">hitB\n"
    "ACDEF---GHIKLMN.QR\n"
)
QUERY_1_UNGAPPED = "ACDEFGHIKLMNPQR"  # gaps ('-') stripped from the first record


def _make_task_dir(tmp_path, a3m_map):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    for name, content in a3m_map.items():
        (task_dir / name).write_text(content)
    return str(task_dir)


# ------------------------------------------------------------------- parse_query_from_a3m
def test_parse_query_strips_gaps_and_skips_comment(tmp_path):
    p = tmp_path / "g.a3m"
    p.write_text(A3M_GROUP_1)
    assert ofu.parse_query_from_a3m(str(p)) == QUERY_1_UNGAPPED


def test_parse_query_takes_only_first_record(tmp_path):
    p = tmp_path / "g.a3m"
    p.write_text(">q\nAAAA\n>other\nCCCCCCCC\n")
    assert ofu.parse_query_from_a3m(str(p)) == "AAAA"


def test_parse_query_strips_dots(tmp_path):
    p = tmp_path / "g.a3m"
    p.write_text(">q\nAC.DE.FG\n")
    assert ofu.parse_query_from_a3m(str(p)) == "ACDEFG"


# ------------------------------------------------------------------ prepare_openfold_input
def test_prepare_openfold_input_layout_and_content(tmp_path):
    task_dir = _make_task_dir(tmp_path, {"group_1.a3m": A3M_GROUP_1})
    work_dir = str(tmp_path / "work")

    fasta_dir, align_dir, template_dir = ofu.prepare_openfold_input(task_dir, work_dir)

    # FASTA: header is the file stem, sequence is the ungapped query
    fasta = os.path.join(fasta_dir, "group_1.fasta")
    assert open(fasta).read() == f">group_1\n{QUERY_1_UNGAPPED}\n"

    # bfd_uniclust_hits.a3m is a VERBATIM copy of the input a3m (all sequences preserved)
    bfd = os.path.join(align_dir, "group_1", "bfd_uniclust_hits.a3m")
    assert open(bfd).read() == A3M_GROUP_1

    # Stockholm stubs contain the query (an empty .sto would crash OpenFold)
    for stub in ("uniref90_hits.sto", "mgnify_hits.sto"):
        sto = os.path.join(align_dir, "group_1", stub)
        assert open(sto).read() == f"# STOCKHOLM 1.0\ngroup_1 {QUERY_1_UNGAPPED}\n//\n"

    # dummy template file required by the CLI parser
    assert os.path.isfile(os.path.join(template_dir, "dummy.cif"))


def test_prepare_openfold_input_multiple_chunks(tmp_path):
    task_dir = _make_task_dir(
        tmp_path, {"a.a3m": ">a\nACDE\n>h\nACDE\n", "b.a3m": ">b\nWYWY\n"}
    )
    work_dir = str(tmp_path / "work")
    fasta_dir, align_dir, _ = ofu.prepare_openfold_input(task_dir, work_dir)
    assert os.path.isfile(os.path.join(fasta_dir, "a.fasta"))
    assert os.path.isfile(os.path.join(fasta_dir, "b.fasta"))
    assert os.path.isdir(os.path.join(align_dir, "a"))
    assert os.path.isdir(os.path.join(align_dir, "b"))


def test_prepare_openfold_input_is_clean_rebuild(tmp_path):
    task_dir = _make_task_dir(tmp_path, {"a.a3m": ">a\nACDE\n"})
    work_dir = str(tmp_path / "work")
    ofu.prepare_openfold_input(task_dir, work_dir)
    # a stale chunk from a previous run must not survive a rebuild
    stale = os.path.join(work_dir, "fasta", "stale.fasta")
    open(stale, "w").write("junk")
    ofu.prepare_openfold_input(task_dir, work_dir)
    assert not os.path.exists(stale)


def test_prepare_openfold_input_no_a3m_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ofu.prepare_openfold_input(str(empty), str(tmp_path / "work"))


# ------------------------------------------------------------------- build_openfold_command
def test_build_command_deepspeed_bf16_flags():
    cmd = ofu.build_openfold_command(
        "fa", "tmpl", "out", "align", openfold_dir="/of",
        openfold_config="deepspeed_bf16", openfold_model="model_3_ptm",
    )
    assert "/of/run_pretrained_openfold.py" in cmd
    assert cmd.split()[2:4] == ["fa", "tmpl"]  # positional fasta_dir, template_dir
    assert "--output_dir out" in cmd
    assert "--use_precomputed_alignments align" in cmd
    assert "--config_preset model_3_ptm" in cmd
    assert "--model_device cuda:0" in cmd
    assert "--skip_relaxation" in cmd
    assert "--use_deepspeed_evoformer_attention" in cmd
    assert "--precision bf16" in cmd
    assert "--save_outputs" not in cmd  # off by default (260 MB pkl/chunk)


def test_build_command_save_outputs_toggle():
    cmd = ofu.build_openfold_command("fa", "t", "o", "a", save_outputs=True)
    assert "--save_outputs" in cmd


def test_build_command_unknown_config_raises():
    with pytest.raises(ValueError):
        ofu.build_openfold_command("fa", "t", "o", "a", openfold_config="nope")


def test_build_command_flash_substitutes_path():
    cmd = ofu.build_openfold_command(
        "fa", "t", "o", "a", openfold_config="bf16_flash",
        flash_config_path="/w/flash_config.json",
    )
    assert "--experiment_config_json /w/flash_config.json" in cmd
    assert "{flash_config}" not in cmd


# ------------------------------------------------------------------- build_openfold_env_setup
def test_env_setup_safety_invariants():
    env = ofu.build_openfold_env_setup("/envs/openfold2", "/of")
    assert "set -eo pipefail" in env
    assert "conda init" not in env          # never: truncates ~/.bashrc under concurrent jobs
    assert "set -u" not in env              # never: conda gcc activate has an unbound var
    assert "conda activate /envs/openfold2" in env
    assert "CUTLASS_PATH=/of/cutlass" in env
    assert env.rstrip().endswith("cd /of")  # must run from the OpenFold source dir


# ------------------------------------------------------------------- collect_openfold_output
def _seed_openfold_pdb(task_dir, tag, model="model_3_ptm", content="ATOM  fake\n"):
    pred_dir = os.path.join(ofu.openfold_output_dir(task_dir), "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, f"{tag}_{model}_unrelaxed.pdb")
    open(path, "w").write(content)
    return path


def test_collect_renames_to_colabfold_convention_with_seed(tmp_path):
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)
    _seed_openfold_pdb(task_dir, "group_1", content="ATOM  X\n")

    n = ofu.collect_openfold_output(task_dir, seed=42)

    assert n == 1
    dest = os.path.join(
        task_dir, "group_1_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
    )
    assert os.path.isfile(dest)
    assert open(dest).read() == "ATOM  X\n"


def test_collect_removes_work_dir_to_avoid_double_count(tmp_path):
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)
    _seed_openfold_pdb(task_dir, "g")
    ofu.collect_openfold_output(task_dir)
    # the raw OpenFold output tree must be gone so rglob('*.pdb') sees only the renamed copy
    assert not os.path.exists(ofu.openfold_work_dir(task_dir))
    remaining = [f for f in os.listdir(task_dir) if f.endswith(".pdb")]
    assert remaining == ["g_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"]


def test_collect_handles_multiple_chunks(tmp_path):
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)
    for tag in ("group_1", "group_2", "group_3"):
        _seed_openfold_pdb(task_dir, tag)
    assert ofu.collect_openfold_output(task_dir) == 3


def test_collect_no_outputs_returns_zero(tmp_path):
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)
    assert ofu.collect_openfold_output(task_dir) == 0


# ------------------------------------------------------------------- write_done_log
def test_write_done_log(tmp_path):
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)
    ofu.write_done_log(task_dir)
    assert "Done" in open(os.path.join(task_dir, "log.txt")).read()
