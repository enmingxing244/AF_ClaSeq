"""Factory for creating job executors based on configuration."""

from typing import Any

from af_claseq.utils.logging_utils import get_logger

logger = get_logger("executor_factory")


def create_executor(
    raw_config: dict,
    **kwargs
) -> Any:
    """Create the appropriate job executor based on config sections.

    Checks for 'slurm' and 'local_gpu' keys in raw_config.
    They are mutually exclusive.

    Args:
        raw_config: Dict that must contain either a 'slurm' or 'local_gpu' key
            with a dict of parameters for the corresponding executor.
        **kwargs: Additional params forwarded to the executor
            (num_models, num_seeds, job_name_prefix, etc.)

    Returns:
        SlurmJobSubmitter or LocalGPUExecutor instance.

    Raises:
        ValueError: If both or neither execution section is present.
    """
    has_slurm = 'slurm' in raw_config and raw_config['slurm'] is not None
    has_local = 'local_gpu' in raw_config and raw_config['local_gpu'] is not None

    if has_slurm and has_local:
        raise ValueError(
            "Config error: Cannot specify both 'slurm' and 'local_gpu' sections. "
            "Please choose one execution mode."
        )

    if not has_slurm and not has_local:
        raise ValueError(
            "Config error: Must specify either 'slurm' or 'local_gpu' section "
            "to define the execution mode."
        )

    if has_slurm:
        from af_claseq.utils.slurm_utils import SlurmJobSubmitter
        logger.info("Using SLURM execution mode")
        return SlurmJobSubmitter(**raw_config['slurm'], **kwargs)

    else:
        from af_claseq.utils.local_gpu_executor import LocalGPUExecutor
        logger.info("Using local GPU execution mode")
        return LocalGPUExecutor(**raw_config['local_gpu'], **kwargs)
