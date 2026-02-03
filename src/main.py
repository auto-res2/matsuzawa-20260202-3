import os
import subprocess
import sys

import hydra


def apply_mode_overrides(cfg) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and hasattr(cfg.run, "optuna"):
            cfg.run.optuna.n_trials = 0
        if hasattr(cfg, "run") and hasattr(cfg.run, "training"):
            cfg.run.training.epochs = 1
            cfg.run.training.max_eval_batches = 2
            cfg.run.training.batch_size = min(int(cfg.run.training.batch_siz


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg) -> None:
    apply_mode_overrides(cfg)

    # Handle case where cfg.run is a string (run_id) instead of config object
    if isinstance(cfg.run, str):
        run_id = cfg.run
    else:
        run_id = cfg.run.run_id
        f"runs@run={run_id}",
        f"results_dir={results_dir}",
        f"mode={cfg.mode}",
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = cfg.wandb.mode
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main
    main()
