import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import OmegaConf

from src.model import (
    PromptEditor,
    PromptModel,
    levenshtein_words,
    normalize_label,
    sanitize_prompt_output,
)
from src.preprocess import DatasetBundle, prepare_datasets

CACHE_DIR = ".cache/"
DEFAULT_PROMPT = (
    "Decide the sentiment of the text. Reply with exactly one word: Positive or Negative."
)
ANSWER_SUFFIX = "Answer with exactly one word: Positive or Negative."


@dataclass
class EvalResult:
    acc: float
    preds: List[int]
    gold: List[int]
    errors: List[Tuple[str, int, str]]
    confusion: Dict[str, int]
    n_samples: int


@dataclass
class RunSummary:
    final_accuracy: float
    final_acc_in: float
    final_acc_ood: float
    final_worst_corpus_accuracy: float
    best_accuracy: float
    best_prompt: str
    accepted_update_count: int
    stream_consumed_in: int
    stream_consumed_ood: int
    prompt_history: List[Dict[str, object]]


@dataclass
class SPRTState:
    wins: int = 0
    losses: int = 0
    llr: float = 0.0


class WandbLogger:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.step = 0

    def log(self, metrics: Dict[str, float]) -> None:
        if self.enabled:
            wandb.log(metrics, step=self.step)
        self.step += 1


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def alpha_spend(alpha_total: float, t: int) -> float:
    return alpha_total * (6.0 / (math.pi**2)) * (1.0 / ((t + 1) ** 2))


def sprt_update(
    state: SPRTState, old_ok: np.ndarray, new_ok: np.ndarray, p0: float, p1: float
) -> SPRTState:
    wins = int(np.logical_and(~old_ok, new_ok).sum())
    losses = int(np.logical_and(old_ok, ~new_ok).sum())
    state.wins += wins
    state.losses += losses
    if wins + losses > 0:
        state.llr = (state.wins * math.log(p1 / p0)) + (
            state.losses * math.log((1 - p1) / (1 - p0))
        )
    return state


def sprt_decide(state: SPRTState, alpha: float, beta: float) -> str:
    alpha = max(alpha, 1e-12)
    beta = max(beta, 1e-12)
    threshold_accept = math.log((1 - beta) / alpha)
    threshold_reject = math.log(beta / (1 - alpha))
    if state.llr >= threshold_accept:
        return "accept"
    if state.llr <= threshold_reject:
        return "reject"
    return "continue"


def evaluate_prompt(
    model: PromptModel,
    prompt: str,
    dataset,
    text_field: str,
    label_field: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    max_batches: Optional[int] = None,
    max_errors: Optional[int] = None,
    logger: Optional[WandbLogger] = None,
    log_prefix: Optional[str] = None,
    log_batch_metrics: bool = False,
) -> EvalResult:
    texts = [ex[text_field] for ex in dataset]
    labels = [int(ex[label_field]) for ex in dataset]
    preds: List[int] = []
    errors: List[Tuple[str, int, str]] = []
    tn = fp = fn = tp = 0

    total_batches = math.ceil(len(texts) / batch_size) if batch_size > 0 else 0
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(len(texts), start + batch_size)
        batch_texts = texts[start:end]
        batch_labels = labels[start:end]
        if batch_idx == 0:
            assert len(batch_texts) == len(
                batch_labels
            ), "Batch inputs and labels must align."

        batch_inputs = [
            f"{prompt}\n\nText: {text}\n{ANSWER_SUFFIX}" for text in batch_texts
        ]
        batch_outputs = model.generate_texts(
            batch_inputs, max_new_tokens=max_new_tokens, max_length=max_length
        )
        if batch_idx == 0:
            assert len(batch_outputs) == len(
                batch_labels
            ), "Model outputs must align with labels."

        batch_preds: List[int] = []
        for text, label, output in zip(batch_texts, batch_labels, batch_outputs):
            pred = normalize_label(output)
            preds.append(pred)
            batch_preds.append(pred)
            if pred != label:
                if max_errors is None or len(errors) < max_errors:
                    errors.append((text, label, output))
            tn += int(pred == 0 and label == 0)
            fp += int(pred == 1 and label == 0)
            fn += int(pred == 0 and label == 1)
            tp += int(pred == 1 and label == 1)

        if logger is not None and log_batch_metrics and log_prefix is not None:
            batch_acc = (
                float((np.array(batch_preds) == np.array(batch_labels)).mean())
                if len(batch_labels) > 0
                else 0.0
            )
            logger.log(
                {
                    f"{log_prefix}_batch_accuracy": batch_acc,
                    f"{log_prefix}_batch_idx": batch_idx,
                    f"{log_prefix}_batch_size": len(batch_labels),
                }
            )

    gold = labels[: len(preds)]
    preds_arr = np.array(preds)
    gold_arr = np.array(gold)
    acc = float((preds_arr == gold_arr).mean()) if len(preds_arr) > 0 else 0.0
    confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    return EvalResult(
        acc=acc,
        preds=preds,
        gold=gold,
        errors=errors,
        confusion=confusion,
        n_samples=len(preds),
    )


def get_stability_params(cfg) -> Tuple[int, int]:
    if "stability" in cfg.method_params:
        max_prompt_chars = int(cfg.method_params.stability.max_prompt_chars)
        max_edit_distance_words = int(cfg.method_params.stability.max_edit_distance_words)
    else:
        max_prompt_chars = int(cfg.method_params.max_prompt_chars)
        max_edit_distance_words = int(cfg.method_params.max_edit_distance_words)
    return max_prompt_chars, max_edit_distance_words


def maybe_train_classifier(
    model: PromptModel,
    datasets: DatasetBundle,
    cfg,
    logger: WandbLogger,
) -> None:
    if cfg.training.learning_rate <= 0 or cfg.training.optimizer == "none":
        return
    if cfg.training.epochs <= 0:
        return

    finetune_backbone = bool(getattr(cfg.training, "finetune_backbone", False))
    if not finetune_backbone:
        model.freeze_backbone()

    model.set_train_mode(train_backbone=finetune_backbone)

    trainable_params = [p for p in model.trainable_parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters available for training.")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    batch_size = int(cfg.training.batch_size)
    max_length = int(cfg.dataset.max_length)
    grad_accum = max(1, int(cfg.training.gradient_accumulation_steps))
    labels = [int(ex[datasets.label_field_in]) for ex in datasets.dev_in]
    texts = [ex[datasets.text_field_in] for ex in datasets.dev_in]

    loss_fn = torch.nn.CrossEntropyLoss()
    step_count = 0

    for epoch in range(int(cfg.training.epochs)):
        indices = np.random.permutation(len(texts))
        for batch_idx, batch_start in enumerate(range(0, len(indices), batch_size)):
            batch_indices = indices[batch_start : batch_start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]
            batch_labels = [labels[i] for i in batch_indices]
            if step_count == 0:
                assert len(batch_texts) == len(
                    batch_labels
                ), "Training batch inputs and labels must align."

            batch_inputs = [
                f"{DEFAULT_PROMPT}\n\nText: {text}\n{ANSWER_SUFFIX}"
                for text in batch_texts
            ]
            logits = model.classify(batch_inputs, max_length=max_length)
            targets = torch.tensor(batch_labels, device=logits.device)

            if step_count == 0:
                assert logits.shape[0] == targets.shape[0], (
                    "Logits and label batch sizes must match."
                )

            loss = loss_fn(logits, targets) / float(grad_accum)
            loss.backward()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                batch_acc = float((preds == targets).float().mean().item())

            if logger.enabled:
                logger.log(
                    {
                        "train_loss": float(loss.item() * grad_accum),
                        "train_acc": batch_acc,
                        "train_epoch": epoch,
                        "train_batch_idx": batch_idx,
                    }
                )

            if (step_count + 1) % grad_accum == 0:
                aux_loss = sum((p.float() ** 2).mean() for p in trainable_params)
                aux_grads = torch.autograd.grad(
                    aux_loss,
                    trainable_params,
                    create_graph=False,
                    retain_graph=True,
                )
                with torch.no_grad():
                    for param, aux_grad in zip(trainable_params, aux_grads):
                        if param.grad is None:
                            param.grad = aux_grad
                        else:
                            param.grad.add_(aux_grad)

                grad_ok = False
                for param in trainable_params:
                    if param.grad is not None and torch.any(param.grad != 0).item():
                        grad_ok = True
                        break
                assert grad_ok, "Gradients missing or zero before optimizer step."

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step_count += 1

    model.set_train_mode(train_backbone=False)


def run_refinement(
    cfg,
    datasets: DatasetBundle,
    classifier: PromptModel,
    editor: PromptEditor,
    logger: WandbLogger,
    log_batch_metrics: bool,
) -> RunSummary:
    num_iterations = int(cfg.method_params.num_iterations)
    proposals_per_iter = int(getattr(cfg.method_params, "proposals_per_iter", 1))
    batch_size = int(cfg.training.batch_size)
    max_length = int(cfg.dataset.max_length)
    max_new_tokens = int(cfg.model.max_new_tokens)
    max_eval_batches = cfg.training.max_eval_batches
    max_prompt_chars, max_edit_distance_words = get_stability_params(cfg)

    prompt = DEFAULT_PROMPT
    best_prompt = prompt
    best_accuracy = -1.0
    accepted_update_count = 0

    pool_in_ptr = 0
    pool_ood_ptr = 0
    stream_consumed_in = 0
    stream_consumed_ood = 0

    prompt_history: List[Dict[str, object]] = []
    method_name = str(cfg.method).lower()

    for iteration in range(num_iterations + 1):
        eval_in = evaluate_prompt(
            classifier,
            prompt,
            datasets.eval_in,
            datasets.text_field_in,
            datasets.label_field_in,
            batch_size,
            max_length,
            max_new_tokens,
            max_batches=max_eval_batches,
            logger=logger,
            log_prefix="eval_in",
            log_batch_metrics=log_batch_metrics,
        )
        eval_ood = evaluate_prompt(
            classifier,
            prompt,
            datasets.eval_ood,
            datasets.text_field_ood,
            datasets.label_field_ood,
            batch_size,
            max_length,
            max_new_tokens,
            max_batches=max_eval_batches,
            logger=logger,
            log_prefix="eval_ood",
            log_batch_metrics=log_batch_metrics,
        )
        balanced_accuracy = 0.5 * (eval_in.acc + eval_ood.acc)
        worst_corpus_accuracy = min(eval_in.acc, eval_ood.acc)
        if balanced_accuracy >= best_accuracy:
            best_accuracy = balanced_accuracy
            best_prompt = prompt

        logger.log(
            {
                "iteration": iteration,
                "eval_accuracy": balanced_accuracy,
                "eval_acc_in": eval_in.acc,
                "eval_acc_ood": eval_ood.acc,
                "worst_corpus_accuracy": worst_corpus_accuracy,
                "accepted_update_count": accepted_update_count,
                "prompt_length": len(prompt),
                "best_eval_accuracy": best_accuracy,
                "stream_consumed_in": stream_consumed_in,
                "stream_consumed_ood": stream_consumed_ood,
            }
        )

        prompt_history.append(
            {
                "iteration": iteration,
                "prompt": prompt,
                "accepted": False,
                "eval_accuracy": balanced_accuracy,
            }
        )

        if iteration == num_iterations:
            break

        if cfg.method_params.acceptance == "fixed":
            continue

        dev_in = evaluate_prompt(
            classifier,
            prompt,
            datasets.dev_in,
            datasets.text_field_in,
            datasets.label_field_in,
            batch_size,
            max_length,
            max_new_tokens,
            max_batches=max_eval_batches,
            max_errors=12,
            logger=logger,
            log_prefix="dev_in",
            log_batch_metrics=log_batch_metrics,
        )
        dev_ood = evaluate_prompt(
            classifier,
            prompt,
            datasets.dev_ood,
            datasets.text_field_ood,
            datasets.label_field_ood,
            batch_size,
            max_length,
            max_new_tokens,
            max_batches=max_eval_batches,
            max_errors=12,
            logger=logger,
            log_prefix="dev_ood",
            log_batch_metrics=log_batch_metrics,
        )
        stats = f"SST-2 acc={dev_in.acc:.3f}, Yelp acc={dev_ood.acc:.3f}"

        for proposal_idx in range(proposals_per_iter):
            if "self-refine" in method_name or "no-perf" in method_name:
                candidate = editor.self_refine_prompt(prompt)
            else:
                candidate = editor.propose_prompt(prompt, stats, dev_in.errors, dev_ood.errors)

            candidate = sanitize_prompt_output(candidate)
            edit_distance = levenshtein_words(prompt, candidate)

            if len(candidate) == 0:
                logger.log(
                    {
                        "iteration": iteration,
                        "proposal_idx": proposal_idx,
                        "candidate_prompt_length": len(candidate),
                        "edit_distance_words": edit_distance,
                        "accepted": 0,
                        "decision_code": -2,
                    }
                )
                continue

            if len(candidate) > max_prompt_chars or edit_distance > max_edit_distance_words:
                logger.log(
                    {
                        "iteration": iteration,
                        "proposal_idx": proposal_idx,
                        "candidate_prompt_length": len(candidate),
                        "edit_distance_words": edit_distance,
                        "accepted": 0,
                        "decision_code": -2,
                    }
                )
                continue

            accept = False
            decision_code = 0
            if cfg.method_params.acceptance == "dev_in_accuracy_non_decrease":
                cand_in = evaluate_prompt(
                    classifier,
                    candidate,
                    datasets.dev_in,
                    datasets.text_field_in,
                    datasets.label_field_in,
                    batch_size,
                    max_length,
                    max_new_tokens,
                    max_batches=max_eval_batches,
                )
                accept = cand_in.acc >= dev_in.acc
                decision_code = 1 if accept else -1
            elif cfg.method_params.acceptance == "mean_dev_accuracy_non_decrease":
                cand_in = evaluate_prompt(
                    classifier,
                    candidate,
                    datasets.dev_in,
                    datasets.text_field_in,
                    datasets.label_field_in,
                    batch_size,
                    max_length,
                    max_new_tokens,
                    max_batches=max_eval_batches,
                )
                cand_ood = evaluate_prompt(
                    classifier,
                    candidate,
                    datasets.dev_ood,
                    datasets.text_field_ood,
                    datasets.label_field_ood,
                    batch_size,
                    max_length,
                    max_new_tokens,
                    max_batches=max_eval_batches,
                )
                accept = 0.5 * (cand_in.acc + cand_ood.acc) >= 0.5 * (
                    dev_in.acc + dev_ood.acc
                )
                decision_code = 1 if accept else -1
            elif cfg.method_params.acceptance == "sequential_mcnemar_sprt":
                if datasets.pool_in is None or datasets.pool_ood is None:
                    raise ValueError("SAFE-Refine requires streaming pools.")
                state = SPRTState()
                alpha_t = alpha_spend(float(cfg.method_params.alpha_total), iteration)
                p1 = 0.5 + float(cfg.method_params.eta)
                decision = "continue"
                max_stream_steps = int(cfg.method_params.max_stream_steps)
                stream_batch = int(cfg.method_params.stream_batch)
                tolerance = float(cfg.method_params.worst_corpus_tolerance)

                for stream_step in range(max_stream_steps):
                    if pool_in_ptr + stream_batch > len(datasets.pool_in):
                        decision = "reject"
                        break
                    if pool_ood_ptr + stream_batch > len(datasets.pool_ood):
                        decision = "reject"
                        break

                    batch_in = datasets.pool_in.select(
                        range(pool_in_ptr, pool_in_ptr + stream_batch)
                    )
                    batch_ood = datasets.pool_ood.select(
                        range(pool_ood_ptr, pool_ood_ptr + stream_batch)
                    )
                    pool_in_ptr += stream_batch
                    pool_ood_ptr += stream_batch
                    stream_consumed_in += stream_batch
                    stream_consumed_ood += stream_batch

                    old_in = evaluate_prompt(
                        classifier,
                        prompt,
                        batch_in,
                        datasets.text_field_in,
                        datasets.label_field_in,
                        batch_size,
                        max_length,
                        max_new_tokens,
                        max_batches=max_eval_batches,
                    )
                    new_in = evaluate_prompt(
                        classifier,
                        candidate,
                        batch_in,
                        datasets.text_field_in,
                        datasets.label_field_in,
                        batch_size,
                        max_length,
                        max_new_tokens,
                        max_batches=max_eval_batches,
                    )
                    old_ood = evaluate_prompt(
                        classifier,
                        prompt,
                        batch_ood,
                        datasets.text_field_ood,
                        datasets.label_field_ood,
                        batch_size,
                        max_length,
                        max_new_tokens,
                        max_batches=max_eval_batches,
                    )
                    new_ood = evaluate_prompt(
                        classifier,
                        candidate,
                        batch_ood,
                        datasets.text_field_ood,
                        datasets.label_field_ood,
                        batch_size,
                        max_length,
                        max_new_tokens,
                        max_batches=max_eval_batches,
                    )

                    old_ok = np.array(old_in.preds + old_ood.preds) == np.array(
                        old_in.gold + old_ood.gold
                    )
                    new_ok = np.array(new_in.preds + new_ood.preds) == np.array(
                        new_in.gold + new_ood.gold
                    )

                    state = sprt_update(state, old_ok, new_ok, p0=0.5, p1=p1)
                    old_worst = min(old_in.acc, old_ood.acc)
                    new_worst = min(new_in.acc, new_ood.acc)

                    if new_worst + tolerance < old_worst:
                        decision = "reject"
                        logger.log(
                            {
                                "iteration": iteration,
                                "proposal_idx": proposal_idx,
                                "stream_step": stream_step,
                                "stream_old_worst": old_worst,
                                "stream_new_worst": new_worst,
                                "stream_decision": -1,
                            }
                        )
                        break

                    decision = sprt_decide(
                        state, alpha=alpha_t, beta=float(cfg.method_params.beta)
                    )
                    stream_decision = 1 if decision == "accept" else (-1 if decision == "reject" else 0)

                    logger.log(
                        {
                            "iteration": iteration,
                            "proposal_idx": proposal_idx,
                            "stream_step": stream_step,
                            "stream_llr": state.llr,
                            "stream_wins": state.wins,
                            "stream_losses": state.losses,
                            "stream_decision": stream_decision,
                            "stream_consumed_in": stream_consumed_in,
                            "stream_consumed_ood": stream_consumed_ood,
                        }
                    )

                    if decision != "continue":
                        break

                accept = decision == "accept"
                decision_code = 1 if accept else -1
            elif cfg.method_params.acceptance == "always":
                accept = True
                decision_code = 1
            else:
                accept = True
                decision_code = 1

            if accept:
                prompt = candidate
                accepted_update_count += 1

            logger.log(
                {
                    "iteration": iteration,
                    "proposal_idx": proposal_idx,
                    "dev_acc_in": dev_in.acc,
                    "dev_acc_ood": dev_ood.acc,
                    "candidate_prompt_length": len(candidate),
                    "edit_distance_words": edit_distance,
                    "accepted": int(accept),
                    "decision_code": decision_code,
                    "accepted_update_count": accepted_update_count,
                    "prompt_length": len(prompt),
                }
            )

            prompt_history.append(
                {
                    "iteration": iteration,
                    "proposal_idx": proposal_idx,
                    "prompt": candidate,
                    "accepted": accept,
                    "edit_distance_words": edit_distance,
                }
            )

            if accept:
                break

    final_eval_in = evaluate_prompt(
        classifier,
        prompt,
        datasets.eval_in,
        datasets.text_field_in,
        datasets.label_field_in,
        batch_size,
        max_length,
        max_new_tokens,
        max_batches=max_eval_batches,
    )
    final_eval_ood = evaluate_prompt(
        classifier,
        prompt,
        datasets.eval_ood,
        datasets.text_field_ood,
        datasets.label_field_ood,
        batch_size,
        max_length,
        max_new_tokens,
        max_batches=max_eval_batches,
    )

    final_accuracy = 0.5 * (final_eval_in.acc + final_eval_ood.acc)
    final_worst = min(final_eval_in.acc, final_eval_ood.acc)

    if logger.enabled:
        wandb.summary["final_accuracy"] = final_accuracy
        wandb.summary["final_acc_in"] = final_eval_in.acc
        wandb.summary["final_acc_ood"] = final_eval_ood.acc
        wandb.summary["final_worst_corpus_accuracy"] = final_worst
        wandb.summary["best_accuracy"] = best_accuracy
        wandb.summary["accepted_update_count"] = accepted_update_count
        wandb.summary["stream_consumed_in"] = stream_consumed_in
        wandb.summary["stream_consumed_ood"] = stream_consumed_ood
        wandb.summary["final_prompt_length"] = len(prompt)
        wandb.summary["best_prompt_length"] = len(best_prompt)
        wandb.summary["confusion_matrix"] = {
            "tn": final_eval_in.confusion["tn"] + final_eval_ood.confusion["tn"],
            "fp": final_eval_in.confusion["fp"] + final_eval_ood.confusion["fp"],
            "fn": final_eval_in.confusion["fn"] + final_eval_ood.confusion["fn"],
            "tp": final_eval_in.confusion["tp"] + final_eval_ood.confusion["tp"],
        }

    return RunSummary(
        final_accuracy=final_accuracy,
        final_acc_in=final_eval_in.acc,
        final_acc_ood=final_eval_ood.acc,
        final_worst_corpus_accuracy=final_worst,
        best_accuracy=best_accuracy,
        best_prompt=best_prompt,
        accepted_update_count=accepted_update_count,
        stream_consumed_in=stream_consumed_in,
        stream_consumed_ood=stream_consumed_ood,
        prompt_history=prompt_history,
    )


def apply_mode_overrides(cfg) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and hasattr(cfg.run, "optuna"):
            cfg.run.optuna.n_trials = 0
        cfg.run.training.epochs = 1
        cfg.run.training.max_eval_batches = 2
        cfg.run.training.batch_size = min(int(cfg.run.training.batch_size), 8)
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


def get_optuna_cfg(run_cfg):
    optuna_cfg = getattr(run_cfg, "optuna", None)
    if optuna_cfg is None:
        return None
    n_trials = int(getattr(optuna_cfg, "n_trials", 0) or 0)
    search_spaces = getattr(optuna_cfg, "search_spaces", None)
    if n_trials <= 0 or not search_spaces:
        return None
    return optuna_cfg


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg) -> None:
    os.environ.setdefault("HF_HOME", CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    apply_mode_overrides(cfg)

    run_cfg = cfg.run
    set_global_seed(int(run_cfg.training.seed))

    datasets = prepare_datasets(run_cfg)

    classifier = PromptModel.from_config(run_cfg.model)
    editor_name = getattr(run_cfg.model, "editor_name", run_cfg.model.name)
    if editor_name == run_cfg.model.name:
        editor = PromptEditor.from_existing(classifier)
    else:
        editor = PromptEditor.from_config(run_cfg.model, editor_name=editor_name)

    assert (
        classifier.tokenizer.pad_token_id is not None
    ), "Tokenizer pad_token_id must be set."
    assert run_cfg.model.max_new_tokens > 0, "max_new_tokens must be positive."
    assert (
        classifier.classifier_head.out_features == 2
    ), "Classifier head output dimension must be 2."
    if classifier.model_type == "seq2seq":
        assert (
            classifier.model.config.is_encoder_decoder
        ), "Seq2seq model must be encoder-decoder."

    optuna_cfg = get_optuna_cfg(run_cfg)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
        if "method_params" not in trial_cfg:
            trial_cfg.method_params = {}
        for space in trial_cfg.optuna.search_spaces:
            name = space.param_name
            if space.distribution_type == "uniform":
                value = trial.suggest_float(name, space.low, space.high)
            elif space.distribution_type == "loguniform":
                value = trial.suggest_float(name, space.low, space.high, log=True)
            elif space.distribution_type == "int":
                value = trial.suggest_int(name, int(space.low), int(space.high))
            elif space.distribution_type == "categorical":
                value = trial.suggest_categorical(name, list(space.choices))
            else:
                raise ValueError(f"Unsupported distribution: {space.distribution_type}")
            trial_cfg.method_params[name] = value
        set_global_seed(int(trial_cfg.training.seed))
        temp_logger = WandbLogger(enabled=False)
        summary = run_refinement(
            trial_cfg,
            datasets,
            classifier,
            editor,
            logger=temp_logger,
            log_batch_metrics=False,
        )
        return summary.final_accuracy

    if optuna_cfg is not None:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(optuna_cfg.n_trials))
        for key, value in study.best_params.items():
            run_cfg.method_params[key] = value

    run = None
    if cfg.wandb.mode != "disabled":
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )

    logger = WandbLogger(enabled=run is not None)
    maybe_train_classifier(classifier, datasets, run_cfg, logger)
    summary = run_refinement(
        run_cfg,
        datasets,
        classifier,
        editor,
        logger=logger,
        log_batch_metrics=run is not None,
    )

    if run is not None:
        print(run.url)
        wandb.finish()

    _ = summary


if __name__ == "__main__":
    main()
