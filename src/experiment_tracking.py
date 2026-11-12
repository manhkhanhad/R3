"""Shared Weights & Biases setup for training and evaluation processes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CODE_SUFFIXES = (".py", ".yaml", ".sbatch", ".sh")
_CODE_METADATA_DIRS = {".git", ".wandb", "wandb"}


def wandb_enabled() -> bool:
    """Return whether W&B logging is enabled for this process."""
    disabled = os.environ.get("WANDB_DISABLED", "").strip().lower()
    if disabled in _TRUE_VALUES:
        return False
    return os.environ.get("WANDB_MODE", "").strip().lower() != "disabled"


def wandb_report_to() -> str:
    """Value to use for Transformers' ``report_to`` argument."""
    return "wandb" if wandb_enabled() else "none"


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _process_label(role: str) -> str:
    scheduler_parts = filter(
        None,
        (
            os.environ.get("SLURM_JOB_ID"),
            os.environ.get("SLURM_ARRAY_TASK_ID"),
            os.environ.get("SLURM_PROCID"),
            os.environ.get("LOCAL_RANK"),
        ),
    )
    suffix = "-".join((*scheduler_parts, str(os.getpid())))
    return f"{role}-{suffix}"


def _tags(dataset: str) -> list[str]:
    tags = [dataset]
    for tag in os.environ.get("WANDB_TAGS", "").split(","):
        tag = tag.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _exclude_code(path: str, root: str) -> bool:
    """Exclude W&B metadata and every path matched by Git ignore rules."""
    relative_path = os.path.relpath(path, root)
    path_parts = Path(relative_path).parts
    if path_parts and path_parts[0] in _CODE_METADATA_DIRS:
        return True
    result = subprocess.run(
        [
            "git",
            "-C",
            root,
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative_path,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Could not evaluate Git ignore rules for {relative_path}")
    return result.returncode == 0


def move_callback_to_end(trainer, callback_type) -> None:
    """Move an existing Trainer callback behind callbacks that enrich logs."""
    callback = trainer.pop_callback(callback_type)
    if callback is not None:
        trainer.add_callback(callback)


def init_wandb_run(
    *,
    dataset: str,
    role: str,
    primary: bool,
    run_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
):
    """Initialize a normal run or join the shared ``WANDB_RUN_ID`` run.

    A supplied ``WANDB_RUN_ID`` opts into W&B shared mode. This allows the
    training and evaluation scripts to write to one run concurrently. Only the
    primary training process is allowed to finalize that shared run.
    """
    if not wandb_enabled():
        return None
    if primary and not _is_main_process():
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed. Install "
            "environment.yaml or set WANDB_DISABLED=true."
        ) from exc

    run_id = os.environ.get("WANDB_RUN_ID") or None
    settings = None
    if run_id:
        settings_kwargs: dict[str, Any] = {
            "mode": "shared",
            "x_label": _process_label(role),
            "x_primary": primary,
        }
        if not primary:
            settings_kwargs["x_update_finish_state"] = False
        settings = wandb.Settings(**settings_kwargs)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "R3"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        id=run_id,
        name=run_name or os.environ.get("WANDB_NAME") or dataset,
        tags=_tags(dataset),
        job_type=None if run_id else role,
        config=dict(config) if config is not None else None,
        settings=settings,
    )
    run.log_code(
        str(_PROJECT_ROOT),
        include_fn=lambda path: path.endswith(_CODE_SUFFIXES),
        exclude_fn=_exclude_code,
    )
    return run


def init_training_run(
    dataset: str,
    run_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
):
    return init_wandb_run(
        dataset=dataset,
        role="training",
        primary=True,
        run_name=run_name,
        config=config,
    )


def init_evaluation_run(
    dataset: str,
    run_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
):
    return init_wandb_run(
        dataset=dataset,
        role="evaluation",
        primary=False,
        run_name=run_name,
        config=config,
    )
