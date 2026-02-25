# Design: Local Multi-GPU Execution Support

**Date:** 2026-02-25
**Branch:** `feature-local-gpu`
**Status:** Approved

## Problem

AF_ClaSeq currently supports only SLURM-based job submission for ColabFold predictions. Users with local multi-GPU machines (no SLURM scheduler) cannot run the pipeline. This feature adds a `local_gpu` execution mode that runs ColabFold jobs locally, pinning one job per GPU via `CUDA_VISIBLE_DEVICES`.

## Design Decisions

1. **Additive strategy** — existing `slurm:` config and `SlurmJobSubmitter` are not modified
2. **Mutual exclusion** — config must have exactly one of `slurm:` or `local_gpu:` section; both present = error
3. **One job per GPU** — each ColabFold job is pinned to a single GPU; N GPUs = N concurrent jobs
4. **Pre-configured environment** — local mode assumes `colabfold_batch` is already in PATH (no `module load`)
5. **Same interface** — `LocalGPUExecutor` exposes the same methods as `SlurmJobSubmitter` so workflows work unchanged

## Configuration

### Existing SLURM Config (unchanged)

```yaml
slurm:
  conda_env_path: "/path/to/colabfold/env"
  slurm_account: "PAA0203"
  slurm_partition: "nextgen"
  slurm_time: "01:00:00"
  slurm_gpus_per_task: 1
  slurm_cpus_per_task: 8
  max_workers: 200
  # ... all existing fields untouched
```

### New Local GPU Config (mutually exclusive with slurm)

```yaml
local_gpu:
  cuda_visible_devices: "0,1,2,3"   # Required: comma-separated GPU IDs
```

- `cuda_visible_devices` (str, required): Which GPUs to use. Parsed to determine GPU count and concurrent job limit.
- ColabFold parameters (`num_models`, `num_seeds`, `num_recycle`, `random_seed`) come from existing config sections (`general:`, `recompile_predict:`, `structure_prediction:`) — not duplicated.

### Validation

If both `slurm:` and `local_gpu:` sections exist in the same YAML file:
```
ConfigError: Cannot specify both 'slurm' and 'local_gpu' sections.
Please choose one execution mode.
```

If neither section exists:
```
ConfigError: Must specify either 'slurm' or 'local_gpu' section.
```

## Architecture

### New Files

#### `src/af_claseq/utils/local_gpu_executor.py`

```
LocalGPUExecutor
├── __init__(cuda_visible_devices, num_recycle=3, **kwargs)
│   ├── Parse GPU string → list of GPU IDs (e.g., "0,1,2,3" → [0,1,2,3])
│   ├── Create gpu_queue (queue.Queue) with GPU IDs as tokens
│   ├── Store ColabFold params from kwargs (same as SlurmJobSubmitter)
│   └── Validate GPU IDs exist via nvidia-smi at startup
│
├── submit_job(task_dir, job_id, job_type=None) → Optional[str]
│   ├── Acquire GPU from gpu_queue (blocks until available)
│   ├── Build colabfold_batch command (same as SlurmJobSubmitter._get_job_config)
│   ├── Run subprocess with env CUDA_VISIBLE_DEVICES=<gpu_id>
│   ├── Return process PID as job identifier
│   └── Release GPU back to queue when process completes
│
├── process_folder(task_dir, job_id, job_type=None) → None
│   ├── Same retry logic as SlurmJobSubmitter.process_folder
│   ├── submit_job → wait → check PDB files → retry if needed
│   └── Reuse _check_pdb_files, _check_log_file, _backup_log_file
│
├── process_folders_concurrently(folders, job_ids, max_workers, job_types=None) → None
│   ├── Cap max_workers to len(gpu_ids) (can't run more jobs than GPUs)
│   ├── ThreadPoolExecutor with capped max_workers
│   └── Each thread: acquire GPU → run colabfold → release GPU
│
├── wait_for_completion(pid) → None
│   └── subprocess.Popen.wait()
│
├── _check_pdb_files(task_dir) → bool  (same logic as SlurmJobSubmitter)
├── _check_log_file(task_dir) → bool   (same logic as SlurmJobSubmitter)
└── _backup_log_file(task_dir, job_id)  (same logic as SlurmJobSubmitter)
```

**GPU pool mechanism:**
- `queue.Queue` initialized with GPU IDs: `[0, 1, 2, 3]`
- Worker thread calls `gpu_queue.get()` to acquire a GPU (blocks if none available)
- After process completes (success or fail), calls `gpu_queue.put(gpu_id)` to release
- Natural backpressure: if 4 GPUs, max 4 concurrent jobs regardless of max_workers

#### `src/af_claseq/utils/executor_factory.py`

```python
def create_executor(config_dict: dict, **kwargs):
    """
    Create the appropriate job executor based on config.

    Args:
        config_dict: Raw YAML config dict (or parsed config object)
        **kwargs: Additional params (num_models, num_seeds, etc.)

    Returns:
        SlurmJobSubmitter or LocalGPUExecutor

    Raises:
        ValueError: If both or neither execution section present
    """
```

### Modified Files

#### Per-workflow config.py files

Each workflow's config.py needs:

1. **New `LocalGPUConfig` dataclass:**
   ```python
   @dataclass
   class LocalGPUConfig:
       cuda_visible_devices: str
   ```

2. **Make `slurm` optional in the top-level config dataclass:**
   - `PipelineConfig.slurm: Optional[SlurmConfig] = None`
   - `PipelineConfig.local_gpu: Optional[LocalGPUConfig] = None`

3. **Add validation in `load_*_config()` function:**
   - Check mutual exclusion of `slurm:` and `local_gpu:`
   - Raise clear error if both or neither present

Files affected:
- `src/af_claseq/m_fold_sampling_voting/config.py` — `PipelineConfig`, `load_pipeline_config()`
- `src/af_claseq/leave_one_out/config.py` — `WorkflowConfig`, `WorkflowConfig.from_yaml()`
- `src/af_claseq/occurrence_voting/config.py` — `OccurrenceVotingConfig`, `OccurrenceVotingConfig.from_yaml()`

#### Per-workflow run scripts

Each run script's `_init_slurm_submitter()` (or equivalent) is replaced with a factory call:

- `scripts/run_m_fold_sampling_voting.py` — `AFClaSeqPipeline._init_slurm_submitter()` → `_init_executor()`
- `scripts/run_occurrence_voting.py` — wherever `SlurmJobSubmitter` is constructed
- `scripts/run_leave_one_out.py` — same
- `scripts/run_divide_and_conquer.py` — same

The factory reads the config and returns either `SlurmJobSubmitter` or `LocalGPUExecutor`. Since both expose the same interface (`submit_job`, `process_folder`, `process_folders_concurrently`), no other code changes needed.

#### New example config

`example/config_examples/local_gpu_m_fold_config.yaml` — complete working example showing `local_gpu:` instead of `slurm:`.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid GPU ID in `cuda_visible_devices` | Fail fast at startup with `nvidia-smi` validation |
| `colabfold_batch` not in PATH | Clear error: "colabfold_batch not found. Ensure it is installed and in your PATH." |
| ColabFold subprocess fails | Same retry logic as SLURM: backup log, resubmit |
| All GPUs busy | Worker thread blocks on `gpu_queue.get()` until one frees (natural backpressure) |
| Both `slurm:` and `local_gpu:` in config | Raise `ValueError` with clear message |

## Data Flow (Local GPU Mode)

```
YAML Config
  ├── general: {num_models, num_seeds, random_seed, ...}
  └── local_gpu: {cuda_visible_devices: "0,1,2,3"}
       │
       ▼
  create_executor(config)
       │
       ▼
  LocalGPUExecutor(cuda_visible_devices="0,1,2,3", num_models=1, ...)
       │
       ▼
  process_folders_concurrently(folders, job_ids, max_workers=4)
       │
       ├── Thread 1: gpu_queue.get() → GPU 0
       │   └── CUDA_VISIBLE_DEVICES=0 colabfold_batch --num-recycle 3 --num-models 1 dir1 dir1
       ├── Thread 2: gpu_queue.get() → GPU 1
       │   └── CUDA_VISIBLE_DEVICES=1 colabfold_batch --num-recycle 3 --num-models 1 dir2 dir2
       ├── Thread 3: gpu_queue.get() → GPU 2
       │   └── CUDA_VISIBLE_DEVICES=2 colabfold_batch --num-recycle 3 --num-models 1 dir3 dir3
       └── Thread 4: gpu_queue.get() → GPU 3
           └── CUDA_VISIBLE_DEVICES=3 colabfold_batch --num-recycle 3 --num-models 1 dir4 dir4

       When Thread 1 finishes → gpu_queue.put(0) → next folder picks up GPU 0
```

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/af_claseq/utils/local_gpu_executor.py` | NEW | `LocalGPUExecutor` class |
| `src/af_claseq/utils/executor_factory.py` | NEW | `create_executor()` factory function |
| `src/af_claseq/m_fold_sampling_voting/config.py` | EDIT | Add `LocalGPUConfig`, make `slurm` optional, add validation |
| `src/af_claseq/leave_one_out/config.py` | EDIT | Same pattern |
| `src/af_claseq/occurrence_voting/config.py` | EDIT | Same pattern |
| `scripts/run_m_fold_sampling_voting.py` | EDIT | Use `create_executor()` via factory |
| `scripts/run_occurrence_voting.py` | EDIT | Same |
| `scripts/run_leave_one_out.py` | EDIT | Same |
| `scripts/run_divide_and_conquer.py` | EDIT | Same |
| `example/config_examples/local_gpu_m_fold_config.yaml` | NEW | Example config |
