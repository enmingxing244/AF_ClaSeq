"""The prediction_engine + openfold options are exposed in every workflow's SLURM config
section, default to ColabFold (backward compatible), and are settable to OpenFold."""


def test_m_fold_slurm_config_engine_fields():
    from af_claseq.m_fold_sampling_voting.config import SlurmConfig
    base = dict(
        conda_env_path="/e", slurm_account="A", slurm_output="/dev/null",
        slurm_error="/dev/null", slurm_nodes=1, slurm_gpus_per_task=1, slurm_tasks=1,
        slurm_cpus_per_task=8, slurm_time="01:00:00", slurm_partition="p", max_workers=10,
    )
    c = SlurmConfig(**base)
    assert c.prediction_engine == "colabfold"
    assert c.openfold_config == "deepspeed_bf16"
    assert c.openfold_model == "model_3_ptm"
    assert c.openfold_conda_env is None and c.openfold_dir is None
    assert SlurmConfig(**base, prediction_engine="openfold").prediction_engine == "openfold"


def test_loo_slurm_config_engine_fields():
    from af_claseq.leave_one_out.config import SlurmConfig
    c = SlurmConfig(conda_env_path="/e", account="A")
    assert c.prediction_engine == "colabfold"
    assert c.openfold_config == "deepspeed_bf16"
    assert SlurmConfig(conda_env_path="/e", account="A",
                       prediction_engine="openfold").prediction_engine == "openfold"


def test_occurrence_slurm_config_engine_fields():
    from af_claseq.occurrence_voting.config import SlurmConfig
    c = SlurmConfig(conda_env_path="/e", account="A")
    assert c.prediction_engine == "colabfold"
    assert c.openfold_model == "model_3_ptm"
    assert c.openfold_conda_env is None


def test_umap_slurm_section_engine_fields():
    from af_claseq.umap_voting.config import SlurmSection
    c = SlurmSection(conda_env_path="/e", account="A")
    assert c.prediction_engine == "colabfold"
    assert c.openfold_conda_env is None
    assert SlurmSection(conda_env_path="/e", account="A",
                        openfold_config="bf16").openfold_config == "bf16"
