"""Read-only preflight validator for AF_ClaSeq YAML configs."""

import json, shutil, sys
from pathlib import Path
from typing import List
import yaml
from af_claseq.utils.logging_utils import get_logger

logger = get_logger("config_validator")

_PATH_KEYS = {
    "source_a3m", "config_file", "structure_analysis_config",
    "config_json", "default_pdb", "a3m_file", "fasttree_binary",
    "m_fold_samp_input_a3m", "conda_env_path", "conda_env",
}
_JSON_CONFIG_KEYS = {"config_file", "structure_analysis_config", "config_json"}
_METRIC_NAME_KEYS = {
    "metric1_name", "metric2_name", "metric_name_1", "metric_name_2",
    "impact_metric_name", "metric_name",
}
_SLURM_ACCOUNT_KEYS = {"slurm_account", "account"}
_SLURM_PARTITION_KEYS = {"slurm_partition", "partition"}


def _collect_values(data, target_keys):
    """Recursively collect string values for *target_keys* from nested data."""
    found = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k in target_keys and isinstance(v, str) and v:
                found.append(v)
            else:
                found.extend(_collect_values(v, target_keys))
    elif isinstance(data, list):
        for item in data:
            found.extend(_collect_values(item, target_keys))
    return found

def _all_metric_names_from_json(json_path: str) -> List[str]:
    """Return every metric name defined in a structure-analysis JSON."""
    with open(json_path, "r") as fh:
        data = json.load(fh)
    names = [c["name"] for c in data.get("filter_criteria", []) if "name" in c]
    names += [c["name"] for c in data.get("composite_metrics", []) if "name" in c]
    return names

def validate_config(yaml_path: str) -> List[str]:
    """Validate an AF_ClaSeq YAML config. Returns list of problems (empty = OK)."""
    issues: List[str] = []
    path = Path(yaml_path)

    # 1. File exists and YAML parses
    if not path.is_file():
        return [f"Config file does not exist: {yaml_path}"]
    try:
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]
    if not isinstance(cfg, dict):
        return ["YAML did not produce a top-level mapping"]

    # 2. Required top-level section ('general' or 'input' for divide-and-conquer)
    if "general" not in cfg and "input" not in cfg:
        issues.append("Missing required top-level section: 'general' (or 'input')")

    # 3. File-path existence and worktree check
    for fp in _collect_values(cfg, _PATH_KEYS):
        if ".worktrees/" in fp:
            issues.append(f"Stale worktree reference in path: {fp}")
        if not Path(fp).exists():
            issues.append(f"Referenced path does not exist: {fp}")

    # 4. JSON metric cross-reference
    json_paths = _collect_values(cfg, _JSON_CONFIG_KEYS)
    metric_names = _collect_values(cfg, _METRIC_NAME_KEYS)
    plotting = cfg.get("plotting", {})
    if isinstance(plotting, dict):
        metric_names += plotting.get("metrics", []) or []
        metric_names += plotting.get("metrics_to_plot", []) or []
    for jp in json_paths:
        if not Path(jp).is_file():
            continue
        available = _all_metric_names_from_json(jp)
        for mn in metric_names:
            if mn and mn not in available:
                issues.append(f"Metric '{mn}' not found in {Path(jp).name}. Available: {available}")

    # 5. External tools
    _TMALIGN_SECTIONS = ("structure_analysis", "m_fold_sampling", "recompile_predict", "leave_one_out")
    if any(s in cfg for s in _TMALIGN_SECTIONS) and shutil.which("TMalign") is None:
        issues.append("TMalign not found on PATH (required by this workflow)")
    if any(s in cfg for s in ("slurm", "colabfold", "structure_prediction")) and shutil.which("colabfold_batch") is None:
        issues.append("colabfold_batch not found on PATH (may need conda activation)")

    # 6. SLURM account / partition non-empty
    for k, v in cfg.items():
        if "slurm" not in k.lower() or not isinstance(v, dict):
            continue
        for key in _SLURM_ACCOUNT_KEYS | _SLURM_PARTITION_KEYS:
            val = v.get(key)
            if val is not None and not str(val).strip():
                issues.append(f"SLURM '{key}' is empty")

    return issues


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(2)

    problems = validate_config(sys.argv[1])
    if problems:
        print(f"Found {len(problems)} issue(s):")
        for i, msg in enumerate(problems, 1):
            print(f"  {i}. {msg}")
        sys.exit(1)
    else:
        print("Config OK -- no issues found.")
