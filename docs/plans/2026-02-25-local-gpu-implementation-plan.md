# Local Multi-GPU Support — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `local_gpu` execution mode so ColabFold jobs can run on local multi-GPU machines without SLURM.

**Architecture:** New `LocalGPUExecutor` class with same interface as `SlurmJobSubmitter`. Factory function selects executor based on config. New `local_gpu:` YAML section is mutually exclusive with `slurm:`.

**Tech Stack:** Python 3.10+, dataclasses, queue.Queue, subprocess, ThreadPoolExecutor

**Design doc:** `docs/plans/2026-02-25-local-gpu-support-design.md`
**GitHub issue:** https://github.com/enmingxing244/AF_ClaSeq/issues/27

---

### Task 1: Create `LocalGPUExecutor` class

**Files:**
- Create: `src/af_claseq/utils/local_gpu_executor.py`

**Step 1: Create the LocalGPUExecutor class**

This class mirrors `SlurmJobSubmitter` (see `src/af_claseq/utils/slurm_utils.py:41-537`) but runs jobs locally with GPU pinning via `CUDA_VISIBLE_DEVICES`.

```python
import os
import subprocess
import time
import shutil
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

from af_claseq.utils.logging_utils import get_logger

logger = get_logger("local_gpu_executor")


class LocalGPUExecutor:
    """Execute ColabFold jobs locally on multi-GPU machines.

    Mirrors the SlurmJobSubmitter interface so all workflows work unchanged.
    Pins one ColabFold job per GPU via CUDA_VISIBLE_DEVICES.
    """

    def __init__(
        self,
        cuda_visible_devices: str,
        num_recycle: int = 3,
        job_name_prefix: str = "fold",
        check_interval: int = 10,
        **kwargs
    ):
        # Parse GPU IDs
        self.gpu_ids = [g.strip() for g in cuda_visible_devices.split(",") if g.strip()]
        if not self.gpu_ids:
            raise ValueError("cuda_visible_devices must contain at least one GPU ID")

        self.num_recycle = num_recycle
        self.job_name_prefix = job_name_prefix
        self.check_interval = check_interval

        # ColabFold mode config — same logic as SlurmJobSubmitter (slurm_utils.py:101-119)
        if any(k.startswith('prediction_') for k in kwargs):
            self.mode = 'pure_pred'
            self.job_configs = {
                'prediction': {
                    'num_models': kwargs.get('prediction_num_model', 1),
                    'num_seeds': kwargs.get('prediction_num_seed', 1)
                },
                'control_prediction': {
                    'num_models': kwargs.get('prediction_num_model', 1),
                    'num_seeds': kwargs.get('prediction_num_seed', 1)
                }
            }
        else:
            self.mode = 'batch_pred'
            self.num_models = kwargs.get('num_models', 1)
            self.num_seeds = kwargs.get('num_seeds', 1)
            self.random_seed = kwargs.get('random_seed')

        # GPU pool for concurrency control
        self.gpu_queue = queue.Queue()
        for gpu_id in self.gpu_ids:
            self.gpu_queue.put(gpu_id)

        logger.info(f"LocalGPUExecutor initialized with {len(self.gpu_ids)} GPUs: {self.gpu_ids}")

    def _get_job_config(self, job_type: Optional[str] = None) -> Dict[str, int]:
        """Get job configuration based on mode and job type.
        Same logic as SlurmJobSubmitter._get_job_config (slurm_utils.py:125-135).
        """
        if self.mode == 'pure_pred':
            if not job_type or job_type not in self.job_configs:
                raise ValueError(f"Invalid job type: {job_type}")
            return self.job_configs[job_type]
        else:
            config = {'num_models': self.num_models, 'num_seeds': self.num_seeds}
            if self.random_seed is not None:
                config['random_seed'] = self.random_seed
            return config

    def submit_job(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> Optional[str]:
        """Submit a local ColabFold job pinned to a single GPU.
        Returns the subprocess PID as a string, or None on failure.
        """
        if not os.path.exists(task_dir):
            logger.error(f"Task directory not found: {task_dir}")
            return None

        config = self._get_job_config(job_type)

        # Build colabfold command — same as slurm_utils.py:146-159
        colabfold_cmd = [
            "colabfold_batch",
            "--num-recycle", str(self.num_recycle),
            "--num-models", str(config['num_models']),
            "--num-seeds", str(config['num_seeds']),
        ]
        if 'random_seed' in config:
            colabfold_cmd.extend(["--random-seed", str(config['random_seed'])])
        colabfold_cmd.extend([task_dir, task_dir])

        # Acquire a GPU from the pool (blocks until one is available)
        gpu_id = self.gpu_queue.get()

        job_type_str = f" ({job_type})" if job_type else ""
        logger.info(f"Running job {job_id}{job_type_str} on GPU {gpu_id}: {task_dir}")

        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            process = subprocess.Popen(
                colabfold_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=task_dir
            )

            # Store gpu_id on the process object so we can release it later
            process._gpu_id = gpu_id
            return str(process.pid)

        except Exception as e:
            logger.error(f"Failed to start job for {task_dir}: {e}")
            self.gpu_queue.put(gpu_id)  # Release GPU on failure
            return None

    def _run_and_wait(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> bool:
        """Run a ColabFold job on a GPU and wait for it to complete.
        Returns True if the process exited successfully.
        """
        if not os.path.exists(task_dir):
            logger.error(f"Task directory not found: {task_dir}")
            return False

        config = self._get_job_config(job_type)

        colabfold_cmd = [
            "colabfold_batch",
            "--num-recycle", str(self.num_recycle),
            "--num-models", str(config['num_models']),
            "--num-seeds", str(config['num_seeds']),
        ]
        if 'random_seed' in config:
            colabfold_cmd.extend(["--random-seed", str(config['random_seed'])])
        colabfold_cmd.extend([task_dir, task_dir])

        gpu_id = self.gpu_queue.get()

        job_type_str = f" ({job_type})" if job_type else ""
        logger.info(f"Running job {job_id}{job_type_str} on GPU {gpu_id}: {task_dir}")

        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            process = subprocess.run(
                colabfold_cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=task_dir
            )

            if process.returncode != 0:
                logger.warning(f"Job {job_id} on GPU {gpu_id} exited with code {process.returncode}")
                if process.stderr:
                    logger.warning(f"stderr: {process.stderr[:500]}")
                return False

            logger.info(f"Job {job_id} completed on GPU {gpu_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to run job for {task_dir}: {e}")
            return False

        finally:
            self.gpu_queue.put(gpu_id)  # Always release GPU

    def process_folder(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> None:
        """Process a single folder with retry logic.
        Same behavior as SlurmJobSubmitter.process_folder (slurm_utils.py:212-229).
        """
        logger.info(f"Processing folder: {task_dir}")

        while not self._check_pdb_files(task_dir):
            success = self._run_and_wait(task_dir, job_id, job_type)

            if self._check_log_file(task_dir):
                logger.info(f"All PDB files generated for {task_dir}")
                break
            else:
                logger.warning(f"PDB files missing in {task_dir}. Retrying.")
                self._backup_log_file(task_dir, job_id)

    def process_folders_concurrently(
        self,
        folders: List[str],
        job_ids: List[str],
        max_workers: int,
        job_types: Optional[List[str]] = None
    ) -> None:
        """Process multiple folders concurrently.
        Same interface as SlurmJobSubmitter.process_folders_concurrently (slurm_utils.py:231-261).
        max_workers is capped to the number of GPUs.
        """
        if not folders or not job_ids:
            logger.error("Empty input lists provided")
            return

        if len(folders) != len(job_ids):
            logger.error("Input lists have different lengths")
            return

        if job_types and len(job_types) != len(folders):
            logger.error("Job types list length doesn't match folders list length")
            return

        # Cap concurrency to number of GPUs
        effective_workers = min(max_workers, len(self.gpu_ids))
        logger.info(
            f"Processing {len(folders)} folders with {effective_workers} workers "
            f"({len(self.gpu_ids)} GPUs available)"
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = []
            for i, (folder, job_id) in enumerate(zip(folders, job_ids)):
                job_type = job_types[i] if job_types else None
                future = executor.submit(self.process_folder, folder, job_id, job_type)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing folder: {str(e)}")

    # --- File checking methods (same as SlurmJobSubmitter) ---

    def _check_pdb_files(self, task_dir: str) -> bool:
        """Check if all required PDB files exist. Same as slurm_utils.py:263-273."""
        pdb_files = [
            os.path.splitext(f)[0] + '.pdb'
            for f in os.listdir(task_dir) if f.endswith('.a3m')
        ]
        missing_files = [f for f in pdb_files if not os.path.exists(os.path.join(task_dir, f))]
        if missing_files:
            logger.debug(f"Missing PDB files in {task_dir}: {missing_files}")
            return False
        return True

    def _check_log_file(self, task_dir: str) -> bool:
        """Check if the log file indicates completion. Same as slurm_utils.py:275-285."""
        log_file = os.path.join(task_dir, 'log.txt')
        if not os.path.exists(log_file):
            return False
        try:
            with open(log_file, 'r') as f:
                return 'Done' in f.read()
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")
            return False

    def _backup_log_file(self, task_dir: str, job_id: str) -> None:
        """Backup the log file before retrying. Same as slurm_utils.py:287-302."""
        log_file = os.path.join(task_dir, 'log.txt')
        if not os.path.exists(log_file):
            return

        backup_dir = os.path.join(task_dir, 'log_backups')
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, f'log_{job_id}.txt')

        try:
            shutil.copy2(log_file, backup_file)
            os.remove(log_file)
            logger.info(f"Backed up log file to {backup_file}")
        except Exception as e:
            logger.error(f"Error backing up log file: {e}")

    # --- Monitoring stubs (for workflows that call these) ---

    def check_job_status(self, job_id: str) -> bool:
        """Stub: local jobs are synchronous, always returns False."""
        return False

    def wait_for_completion(self, job_id: str) -> None:
        """Stub: local jobs complete synchronously in _run_and_wait."""
        pass

    def monitor_jobs(
        self,
        job_ids: List[str],
        check_interval: float = 60.0,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Stub: return all jobs as COMPLETED since local jobs are synchronous."""
        from af_claseq.utils.slurm_utils import JobState
        return {jid: JobState.COMPLETED for jid in job_ids}
```

**Step 2: Verify the file was created correctly**

Run: `python3 -c "from af_claseq.utils.local_gpu_executor import LocalGPUExecutor; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/af_claseq/utils/local_gpu_executor.py
git commit -m "feat: add LocalGPUExecutor class for local multi-GPU execution"
```

---

### Task 2: Create executor factory

**Files:**
- Create: `src/af_claseq/utils/executor_factory.py`

**Step 1: Create the factory module**

```python
"""Factory for creating job executors based on configuration."""

from typing import Any, Union

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
        raw_config: The raw YAML config dict (must contain either 'slurm' or 'local_gpu' key).
        **kwargs: Additional params forwarded to the executor (num_models, num_seeds, etc.)

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
```

**Step 2: Verify import**

Run: `python3 -c "from af_claseq.utils.executor_factory import create_executor; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/af_claseq/utils/executor_factory.py
git commit -m "feat: add executor factory for SLURM/local GPU selection"
```

---

### Task 3: Update M-fold config.py

**Files:**
- Modify: `src/af_claseq/m_fold_sampling_voting/config.py:34-47` (add LocalGPUConfig after SlurmConfig)
- Modify: `src/af_claseq/m_fold_sampling_voting/config.py:134-143` (make slurm optional in PipelineConfig)
- Modify: `src/af_claseq/m_fold_sampling_voting/config.py:231-267` (update load_pipeline_config)

**Step 1: Add LocalGPUConfig dataclass**

After line 47 (after `SlurmConfig`), add:

```python
@dataclass
class LocalGPUConfig:
    """Local GPU execution configuration"""
    cuda_visible_devices: str
```

**Step 2: Make slurm optional and add local_gpu in PipelineConfig**

Change `PipelineConfig` (line 134-143) from:

```python
@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    general: GeneralConfig
    slurm: SlurmConfig
    pipeline_control: PipelineControlConfig
    ...
```

to:

```python
@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    general: GeneralConfig
    pipeline_control: PipelineControlConfig
    m_fold_sampling: MFoldSamplingConfig
    sequence_voting: SequenceVotingConfig
    recompile_predict: RecompilePredictConfig
    pure_sequence_plotting: PureSequencePlottingConfig
    slurm: Optional[SlurmConfig] = None
    local_gpu: Optional[LocalGPUConfig] = None
```

**Step 3: Update load_pipeline_config**

Change `load_pipeline_config` (line 231-267) to handle mutual exclusion:

```python
def load_pipeline_config(yaml_input: str) -> PipelineConfig:
    with open(yaml_input, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Validate execution mode: exactly one of slurm or local_gpu
    has_slurm = 'slurm' in yaml_config and yaml_config['slurm'] is not None
    has_local_gpu = 'local_gpu' in yaml_config and yaml_config['local_gpu'] is not None

    if has_slurm and has_local_gpu:
        raise ValueError(
            "Config error: Cannot specify both 'slurm' and 'local_gpu' sections. "
            "Please choose one execution mode."
        )
    if not has_slurm and not has_local_gpu:
        raise ValueError(
            "Config error: Must specify either 'slurm' or 'local_gpu' section "
            "to define the execution mode."
        )

    general_config = GeneralConfig(**yaml_config.get('general', {}))

    if general_config.metric1_name or general_config.metric2_name:
        validate_metric_names(general_config)

    slurm_config = SlurmConfig(**yaml_config['slurm']) if has_slurm else None
    local_gpu_config = LocalGPUConfig(**yaml_config['local_gpu']) if has_local_gpu else None
    pipeline_control_config = PipelineControlConfig(**yaml_config.get('pipeline_control', {}))
    m_fold_sampling_config = MFoldSamplingConfig(**yaml_config.get('m_fold_sampling', {}))
    sequence_voting_config = SequenceVotingConfig(**yaml_config.get('sequence_voting', {}))
    recompile_predict_config = RecompilePredictConfig(**yaml_config.get('recompile_predict', {}))
    pure_sequence_plotting_config = PureSequencePlottingConfig(**yaml_config.get('pure_sequence_plotting', {}))

    return PipelineConfig(
        general=general_config,
        slurm=slurm_config,
        local_gpu=local_gpu_config,
        pipeline_control=pipeline_control_config,
        m_fold_sampling=m_fold_sampling_config,
        sequence_voting=sequence_voting_config,
        recompile_predict=recompile_predict_config,
        pure_sequence_plotting=pure_sequence_plotting_config
    )
```

**Step 4: Commit**

```bash
git add src/af_claseq/m_fold_sampling_voting/config.py
git commit -m "feat: add LocalGPUConfig to m_fold config, make slurm optional"
```

---

### Task 4: Update `run_m_fold_sampling_voting.py`

**Files:**
- Modify: `scripts/run_m_fold_sampling_voting.py:17` (import)
- Modify: `scripts/run_m_fold_sampling_voting.py:44` (rename slurm_submitter → executor)
- Modify: `scripts/run_m_fold_sampling_voting.py:74-91` (replace `_init_slurm_submitter` with `_init_executor`)
- Modify: `scripts/run_m_fold_sampling_voting.py:280` (max_workers access)
- Modify: `scripts/run_m_fold_sampling_voting.py:288` (executor call)
- Modify: `scripts/run_m_fold_sampling_voting.py:673-692` (prediction_config dict for recompile stage)

**Step 1: Update import (line 17)**

Change:
```python
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
```
to:
```python
from af_claseq.utils.executor_factory import create_executor
```

**Step 2: Update `__init__` (line 44)**

Change:
```python
self.slurm_submitter = self._init_slurm_submitter()
```
to:
```python
self.executor = self._init_executor()
```

**Step 3: Replace `_init_slurm_submitter` method (lines 74-91)**

Replace the entire method with:

```python
    def _init_executor(self):
        """Initialize job executor (SLURM or local GPU) from configuration"""
        # Build the raw config dict for the factory
        raw_config = {}
        if self.config.slurm is not None:
            raw_config['slurm'] = {
                'conda_env_path': self.config.slurm.conda_env_path,
                'slurm_account': self.config.slurm.slurm_account,
                'slurm_output': self.config.slurm.slurm_output,
                'slurm_error': self.config.slurm.slurm_error,
                'slurm_nodes': self.config.slurm.slurm_nodes,
                'slurm_gpus_per_task': self.config.slurm.slurm_gpus_per_task,
                'slurm_tasks': self.config.slurm.slurm_tasks,
                'slurm_cpus_per_task': self.config.slurm.slurm_cpus_per_task,
                'slurm_time': self.config.slurm.slurm_time,
                'slurm_partition': self.config.slurm.slurm_partition,
            }
        if self.config.local_gpu is not None:
            raw_config['local_gpu'] = {
                'cuda_visible_devices': self.config.local_gpu.cuda_visible_devices,
            }
        return create_executor(
            raw_config,
            check_interval=self.config.pipeline_control.check_interval,
            job_name_prefix=self.config.general.protein_name,
            num_models=self.config.general.num_models,
            random_seed=self.config.general.random_seed
        )
```

**Step 4: Update all references from `self.slurm_submitter` to `self.executor`**

Search and replace all occurrences of `self.slurm_submitter` with `self.executor` in the file.

**Step 5: Update max_workers access (line 280)**

Change:
```python
max_workers = self.config.slurm.max_workers
```
to:
```python
if self.config.slurm is not None:
    max_workers = self.config.slurm.max_workers
else:
    # Local GPU mode: use number of GPUs as max_workers
    gpu_count = len(self.config.local_gpu.cuda_visible_devices.split(","))
    max_workers = gpu_count
```

**Step 6: Update prediction_config dict (lines 673-692)**

The `run_recompile_and_predict` method builds a dict for `PureSequenceAF2Prediction`.
Change the SLURM-specific keys to be conditional:

```python
                prediction_config = {
                    'pure_seq_pred_base_dir': criterion_output_dir,
                    'bin_numbers': bin_numbers,
                    'combine_bins': self.config.recompile_predict.combine_bins,
                    'prediction_num_model': self.config.recompile_predict.prediction_num_model,
                    'prediction_num_seed': self.config.recompile_predict.prediction_num_seed,
                    'check_interval': self.config.pipeline_control.check_interval,
                    'job_name_prefix': f"{self.config.general.protein_name}_{criterion_name}"
                }

                if self.config.slurm is not None:
                    prediction_config.update({
                        'conda_env_path': self.config.slurm.conda_env_path,
                        'slurm_account': self.config.slurm.slurm_account,
                        'slurm_output': self.config.slurm.slurm_output,
                        'slurm_error': self.config.slurm.slurm_error,
                        'slurm_nodes': self.config.slurm.slurm_nodes,
                        'slurm_gpus_per_task': self.config.slurm.slurm_gpus_per_task,
                        'slurm_tasks': self.config.slurm.slurm_tasks,
                        'slurm_cpus_per_task': self.config.slurm.slurm_cpus_per_task,
                        'slurm_time': self.config.slurm.slurm_time,
                        'slurm_partition': self.config.slurm.slurm_partition,
                        'max_workers': self.config.slurm.max_workers,
                    })
                elif self.config.local_gpu is not None:
                    prediction_config['cuda_visible_devices'] = self.config.local_gpu.cuda_visible_devices
```

**Step 7: Commit**

```bash
git add scripts/run_m_fold_sampling_voting.py
git commit -m "feat: update m_fold pipeline to use executor factory"
```

---

### Task 5: Update `pure_seq_pred.py` to support local GPU

**Files:**
- Modify: `src/af_claseq/m_fold_sampling_voting/pure_seq_pred.py:14` (import)
- Modify: `src/af_claseq/m_fold_sampling_voting/pure_seq_pred.py:38` (rename)
- Modify: `src/af_claseq/m_fold_sampling_voting/pure_seq_pred.py:59-89` (replace `_init_slurm_submitter`)

**Step 1: Update import (line 14)**

Change:
```python
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
```
to:
```python
from af_claseq.utils.executor_factory import create_executor
```

**Step 2: Update init reference (line 38)**

Change `self.submitter = self._init_slurm_submitter()` to `self.submitter = self._init_executor()`.

**Step 3: Replace _init_slurm_submitter (lines 59-89)**

```python
    def _init_executor(self):
        """Initialize the job executor from configuration options."""
        job_prefix = self.config.get('job_name_prefix')
        if not job_prefix:
            base_dir = Path(self.config['pure_seq_pred_base_dir'])
            try:
                base_path_parts = base_dir.parts
                results_idx = list(base_path_parts).index('results')
                job_prefix = base_path_parts[results_idx + 1] if results_idx + 1 < len(base_path_parts) else "fold"
            except (ValueError, AttributeError, IndexError):
                job_prefix = "fold"
                self.logger.warning("Could not extract job prefix, using default: 'fold'")

        # Determine execution mode from config keys
        raw_config = {}
        if 'cuda_visible_devices' in self.config:
            raw_config['local_gpu'] = {
                'cuda_visible_devices': self.config['cuda_visible_devices'],
            }
        else:
            raw_config['slurm'] = {
                'conda_env_path': self.config['conda_env_path'],
                'slurm_account': self.config['slurm_account'],
                'slurm_output': self.config['slurm_output'],
                'slurm_error': self.config['slurm_error'],
                'slurm_nodes': self.config['slurm_nodes'],
                'slurm_gpus_per_task': self.config['slurm_gpus_per_task'],
                'slurm_tasks': self.config['slurm_tasks'],
                'slurm_cpus_per_task': self.config['slurm_cpus_per_task'],
                'slurm_time': self.config['slurm_time'],
                'slurm_partition': self.config['slurm_partition'],
            }

        return create_executor(
            raw_config,
            check_interval=self.config.get('check_interval', 60),
            job_name_prefix=job_prefix,
            prediction_num_model=self.config['prediction_num_model'],
            prediction_num_seed=self.config['prediction_num_seed']
        )
```

**Step 4: Update max_workers usage**

Find where `self.config['max_workers']` is used in `pure_seq_pred.py` and make it conditional:
```python
max_workers = self.config.get('max_workers', len(self.config.get('cuda_visible_devices', '0').split(',')))
```

**Step 5: Commit**

```bash
git add src/af_claseq/m_fold_sampling_voting/pure_seq_pred.py
git commit -m "feat: update pure_seq_pred to use executor factory"
```

---

### Task 6: Update Leave-One-Out config and workflow

**Files:**
- Modify: `src/af_claseq/leave_one_out/config.py:73-79` (WorkflowConfig)
- Modify: `src/af_claseq/leave_one_out/config.py:82-124` (from_yaml)
- Modify: `src/af_claseq/leave_one_out/loo_workflow.py:23` (import)
- Modify: `src/af_claseq/leave_one_out/loo_workflow.py:214-230` (_initialize_job_submitter)

**Step 1: Add LocalGPUConfig to LOO config.py (after line 61, after SlurmConfig)**

```python
@dataclass
class LocalGPUConfig:
    """Local GPU execution configuration"""
    cuda_visible_devices: str
```

**Step 2: Update WorkflowConfig (line 73-79)**

```python
@dataclass
class WorkflowConfig:
    """Complete workflow configuration"""
    general: GeneralConfig
    leave_one_out: LeaveOneOutConfig
    slurm: Optional[SlurmConfig] = None
    local_gpu: Optional[LocalGPUConfig] = None
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
```

**Step 3: Update from_yaml to validate mutual exclusion (lines 82-124)**

In the `from_yaml` method, replace the `required_sections` check and config creation:

```python
        # Validate execution mode
        has_slurm = 'slurm' in config_data and config_data['slurm'] is not None
        has_local_gpu = 'local_gpu' in config_data and config_data['local_gpu'] is not None
        if has_slurm and has_local_gpu:
            raise ValueError("Config error: Cannot specify both 'slurm' and 'local_gpu' sections.")
        if not has_slurm and not has_local_gpu:
            raise ValueError("Config error: Must specify either 'slurm' or 'local_gpu' section.")

        # Validate other required sections
        for section in ['general', 'leave_one_out']:
            if section not in config_data:
                raise ValueError(f"Missing required configuration section: {section}")

        general_config = GeneralConfig(**config_data['general'])
        loo_config = LeaveOneOutConfig(**config_data['leave_one_out'])
        slurm_config = SlurmConfig(**config_data['slurm']) if has_slurm else None
        local_gpu_config = LocalGPUConfig(**config_data['local_gpu']) if has_local_gpu else None

        plotting_data = config_data.get('plotting', {})
        plotting_config = PlottingConfig(**plotting_data)
        if plotting_config.output_dir is None:
            plotting_config.output_dir = str(Path(general_config.base_dir) / "plots")

        workflow_config = cls(
            general=general_config,
            leave_one_out=loo_config,
            slurm=slurm_config,
            local_gpu=local_gpu_config,
            plotting=plotting_config
        )
        workflow_config._validate_paths()
        logger.info("Configuration loaded successfully")
        return workflow_config
```

**Step 4: Update loo_workflow.py import (line 23)**

Change:
```python
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
```
to:
```python
from af_claseq.utils.executor_factory import create_executor
```

**Step 5: Replace `_initialize_job_submitter` (lines 214-230)**

```python
    def _initialize_job_submitter(self):
        """Initialize job executor (SLURM or local GPU) from configuration"""
        slurm_config = self.config.slurm
        raw_config = {}
        if slurm_config is not None:
            raw_config['slurm'] = {
                'conda_env_path': slurm_config.conda_env_path,
                'slurm_account': slurm_config.account,
                'slurm_partition': slurm_config.partition,
                'slurm_time': slurm_config.time,
                'slurm_cpus_per_task': slurm_config.cpus,
            }
        if self.config.local_gpu is not None:
            raw_config['local_gpu'] = {
                'cuda_visible_devices': self.config.local_gpu.cuda_visible_devices,
            }
        return create_executor(
            raw_config,
            num_models=slurm_config.num_models if slurm_config else 5,
            num_seeds=slurm_config.num_seeds if slurm_config else 1,
            job_name_prefix="loo"
        )
```

**Step 6: Commit**

```bash
git add src/af_claseq/leave_one_out/config.py src/af_claseq/leave_one_out/loo_workflow.py
git commit -m "feat: update leave-one-out workflow to support local GPU mode"
```

---

### Task 7: Update Occurrence Voting config and workflow

**Files:**
- Modify: `src/af_claseq/occurrence_voting/config.py:124-134` (add LocalGPUConfig, update OccurrenceVotingConfig)
- Modify: `src/af_claseq/occurrence_voting/config.py:148-206` (update from_yaml)
- Modify: `src/af_claseq/occurrence_voting/occurrence_voting.py:22` (import)
- Modify: `src/af_claseq/occurrence_voting/occurrence_voting.py:71-89` (_initialize_job_submitter)

**Step 1: Add LocalGPUConfig (after SlurmConfig, before OccurrenceVotingConfig)**

```python
@dataclass
class LocalGPUConfig:
    """Local GPU execution configuration"""
    cuda_visible_devices: str
```

**Step 2: Update OccurrenceVotingConfig**

```python
@dataclass
class OccurrenceVotingConfig:
    """Complete occurrence voting workflow configuration"""
    general: GeneralConfig
    sampling: SamplingConfig
    structure_prediction: StructurePredictionConfig
    structure_analysis: StructureAnalysisConfig
    filtering: FilteringConfig
    voting: VotingConfig
    slurm: Optional[SlurmConfig] = None
    local_gpu: Optional[LocalGPUConfig] = None
    plotting: PlottingConfig = None
```

**Step 3: Update from_yaml with mutual exclusion check**

Same pattern as LOO: validate `slurm` vs `local_gpu`, remove `'slurm'` from required_sections, parse conditionally.

**Step 4: Update occurrence_voting.py import and _initialize_job_submitter**

Same pattern as LOO:
```python
from af_claseq.utils.executor_factory import create_executor
```

Replace `_initialize_job_submitter` to use factory with conditional slurm/local_gpu config dict.

**Step 5: Commit**

```bash
git add src/af_claseq/occurrence_voting/config.py src/af_claseq/occurrence_voting/occurrence_voting.py
git commit -m "feat: update occurrence voting to support local GPU mode"
```

---

### Task 8: Update Divide-and-Conquer script

**Files:**
- Modify: `scripts/run_divide_and_conquer.py:29` (import)
- Modify: `scripts/run_divide_and_conquer.py:110-121` (SlurmJobSubmitter construction)

**Step 1: Update import**

Change:
```python
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
```
to:
```python
from af_claseq.utils.executor_factory import create_executor
```

**Step 2: Replace direct SlurmJobSubmitter construction (lines 110-121)**

The divide-and-conquer script reads config as a raw dict. Replace:
```python
slurm_submitter = SlurmJobSubmitter(
    conda_env_path=colabfold_config.get('conda_env', 'colabfold'),
    slurm_account=slurm_config.get('account', 'PAA0203'),
    ...
)
```
with:
```python
raw_config = {}
if slurm_config:
    raw_config['slurm'] = {
        'conda_env_path': colabfold_config.get('conda_env', 'colabfold'),
        'slurm_account': slurm_config.get('account', 'PAA0203'),
        'slurm_partition': slurm_config.get('partition', 'nextgen'),
        'slurm_time': slurm_config.get('time', '00:30:00'),
        'slurm_cpus_per_task': slurm_config.get('cpus', 8),
    }

local_gpu_config = self.config.get('local_gpu', {})
if local_gpu_config:
    raw_config['local_gpu'] = local_gpu_config

executor = create_executor(
    raw_config,
    job_name_prefix="cf",
    num_models=colabfold_config.get('num_models', 1),
    num_seeds=colabfold_config.get('num_seeds', 1),
    num_recycle=colabfold_config.get('num_recycle', 3)
)
```

And rename `slurm_submitter` → `executor` in the rest of that function.

**Step 3: Commit**

```bash
git add scripts/run_divide_and_conquer.py
git commit -m "feat: update divide-and-conquer to support local GPU mode"
```

---

### Task 9: Create example config

**Files:**
- Create: `example/config_examples/local_gpu_m_fold_config.yaml`

**Step 1: Create the example config file**

Copy the structure from `example/config_examples/m_fold_config.yaml` but replace `slurm:` with `local_gpu:`:

```yaml
# ============================================================================
# LOCAL GPU M-FOLD SAMPLING CONFIGURATION EXAMPLE
# ============================================================================
# This config is for running on a local multi-GPU machine WITHOUT SLURM.
# The 'local_gpu' section replaces the 'slurm' section.
# They are mutually exclusive — you cannot have both.
# ============================================================================

general:
  source_a3m: "/path/to/your/alignment.a3m"
  base_dir: "/path/to/output"
  config_file: "/path/to/config.json"
  protein_name: "YourProtein"
  coverage_threshold: 0.8
  num_models: 1
  random_seed: 42
  num_bins: 30

# ============================================================================
# LOCAL GPU CONFIGURATION
# ============================================================================
# Specify which GPUs to use via CUDA_VISIBLE_DEVICES.
# One ColabFold job runs per GPU concurrently.
# Assumes colabfold_batch is already in PATH.
# ============================================================================
local_gpu:
  cuda_visible_devices: "0,1,2,3"

# ============================================================================
# PIPELINE CONTROL
# ============================================================================
pipeline_control:
  stages:
    - "01_M_FOLD_SAMPLING_RUN"
    - "01_M_FOLD_SAMPLING_PLOT"
    - "02_VOTING_RUN"
    - "03_RECOMPILE_PREDICT_RUN"
    - "04_PURE_SEQ_PLOT_RUN"
  check_interval: 10

# ============================================================================
# STAGE 01: M-FOLD SAMPLING CONFIGURATION
# ============================================================================
m_fold_sampling:
  m_fold_samp_input_a3m: "/path/to/your/alignment.a3m"
  m_fold_group_size: 10
  rounds: 1

# ============================================================================
# STAGE 02: SEQUENCE VOTING CONFIGURATION
# ============================================================================
sequence_voting:
  vote_threshold: 0
  use_focused_bins: false

# ============================================================================
# STAGE 03: RECOMPILE & PREDICTION CONFIGURATION
# ============================================================================
recompile_predict:
  bin_numbers_1: [26]
  prediction_num_model: 5
  prediction_num_seed: 8

# ============================================================================
# STAGE 04: PURE SEQUENCE PLOTTING CONFIGURATION
# ============================================================================
pure_sequence_plotting:
  plddt_threshold: 70
  figsize: [15, 7]
  dpi: 300
  max_workers: 8
```

**Step 2: Commit**

```bash
git add example/config_examples/local_gpu_m_fold_config.yaml
git commit -m "docs: add local GPU config example for m-fold sampling"
```

---

### Task 10: Final integration verification

**Step 1: Verify all imports work**

```bash
python3 -c "
from af_claseq.utils.local_gpu_executor import LocalGPUExecutor
from af_claseq.utils.executor_factory import create_executor
from af_claseq.m_fold_sampling_voting.config import load_pipeline_config, LocalGPUConfig
from af_claseq.leave_one_out.config import LocalGPUConfig as LOOLocalGPUConfig
from af_claseq.occurrence_voting.config import LocalGPUConfig as OVLocalGPUConfig
print('All imports OK')
"
```

**Step 2: Verify config loading with local_gpu section**

```bash
python3 -c "
from af_claseq.m_fold_sampling_voting.config import load_pipeline_config
# This should fail with 'source_a3m' error since paths don't exist,
# but it should NOT fail on the local_gpu section parsing
try:
    config = load_pipeline_config('example/config_examples/local_gpu_m_fold_config.yaml')
except Exception as e:
    print(f'Expected error (paths not set): {type(e).__name__}: {e}')
    if 'slurm' in str(e).lower() and 'local_gpu' in str(e).lower():
        print('FAIL: mutual exclusion error on valid config')
    else:
        print('OK: config parsing logic works')
"
```

**Step 3: Verify mutual exclusion**

```bash
python3 -c "
import yaml
from af_claseq.m_fold_sampling_voting.config import load_pipeline_config
import tempfile, os

# Create a YAML with both sections
bad_config = '''
general:
  source_a3m: /tmp/test.a3m
  base_dir: /tmp/test
  config_file: /tmp/test.json
  protein_name: test
slurm:
  conda_env_path: /tmp
  slurm_account: test
  slurm_output: /dev/null
  slurm_error: /dev/null
  slurm_nodes: 1
  slurm_gpus_per_task: 1
  slurm_tasks: 1
  slurm_cpus_per_task: 4
  slurm_time: '01:00:00'
  slurm_partition: test
  max_workers: 1
local_gpu:
  cuda_visible_devices: '0'
m_fold_sampling:
  m_fold_samp_input_a3m: /tmp/test.a3m
'''
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write(bad_config)
    tmp_path = f.name

try:
    load_pipeline_config(tmp_path)
    print('FAIL: should have raised ValueError')
except ValueError as e:
    if 'Cannot specify both' in str(e):
        print('OK: mutual exclusion correctly enforced')
    else:
        print(f'FAIL: wrong error: {e}')
finally:
    os.unlink(tmp_path)
"
```

**Step 4: Final commit with all changes**

```bash
git add -A
git status
# If any unstaged changes remain, add them
git commit -m "feat: complete local multi-GPU execution support (closes #27)"
```
