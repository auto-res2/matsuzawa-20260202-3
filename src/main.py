import os
import subprocess
import sys

import hydra


def apply_mode_overrides(cfg) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and hasattr(cfg.run, "optuna"):
            cfg.run.optuna.n_trials = 0
        cfg.run.training.epochs = 1
        cfg.run.training.max_eval_batches = 2
        cfg.run.training.batch_size = min(int(cfg.run.training.batch_size), 8)
        if "num_iterations" in cfg.run.method_params:
            cfg.run.method_params.num_iterations = min(
                1, int(cfg.run.method_params.num_iterations)
            )
        if "max_stream_steps" in cfg.run.method_params:
            cfg.run.method_params.max_stream_steps = min(
                1, int(cfg.run.method_params.max_stream_steps)
            )
        if "stream_batch" in cfg.run.method_params:
            cfg.run.method_params.stream_batch = min(
                40, int(cfg.run.method_params.stream_batch)
            )
        splits = cfg.run.dataset.splits
        for key, limit in [
            ("pool_train", 200),
            ("dev", 64),
            ("dev_in", 64),
            ("dev_ood", 64),
            ("eval", 128),
        ]:
            if key in splits and splits[key] is not None:
                splits[key] = int(min(int(splits[key]), limit))
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg) -> None:
    apply_mode_overrides(cfg)

    run_id = cfg.run.run_id
    results_dir = cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"runs@run={run_id}",
        f"results_dir={results_dir}",
        f"mode={cfg.mode}",
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = cfg.wandb.mode
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
