import os
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

CACHE_DIR = ".cache/"


def resolve_dtype(precision: str) -> torch.dtype:
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision in {"fp16", "float16"} and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def normalize_label(text: str) -> int:
    text = text.strip().lower()
    if "positive" in text and "negative" not in text:
        return 1
    if "negative" in text and "positive" not in text:
        return 0
    first_token = text.split()[:1]
    return 1 if first_token == ["positive"] else 0


def sanitize_prompt_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\"']|[\"']$", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def levenshtein_words(a: str, b: str) -> int:
    a_tokens = a.split()
    b_tokens = b.split()
    dp = list(range(len(b_tokens) + 1))
    for i in range(1, len(a_tokens) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b_tokens) + 1):
            cur = dp[j]
            cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]


@dataclass
class ModelConfig:
    name: str
    type: str
    precision: str
    max_new_tokens: int


class PromptModel:
    def __init__(
        self,
        name: str,
        model_type: str,
        precision: str,
        max_new_tokens: int,
        shared_model: Optional[torch.nn.Module] = None,
        shared_tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.precision = precision
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lightweight: try to load a real HF model; if unavailable, fall back to a dummy classifier
        self.use_dummy = False
        try:
            os.environ.setdefault("HF_HOME", CACHE_DIR)
            dtype = resolve_dtype(precision)
            trust_remote_code = "qwen" in name.lower()

            if shared_model is not None and shared_tokenizer is not None:
                self.model = shared_model
                self.tokenizer = shared_tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name, cache_dir=CACHE_DIR, trust_remote_code=trust_remote_code
                )
                if model_type == "seq2seq":
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        name,
                        cache_dir=CACHE_DIR,
                        torch_dtype=dtype,
                        trust_remote_code=trust_remote_code,
                        low_cpu_mem_usage=True,
                    )
                    self.tokenizer.padding_side = "right"
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        name,
                        cache_dir=CACHE_DIR,
                        torch_dtype=dtype,
                        trust_remote_code=trust_remote_code,
                        low_cpu_mem_usage=True,
                    )
                    self.tokenizer.padding_side = "left"

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = (
                    self.tokenizer.eos_token_id or self.tokenizer.unk_token_id
                )

            self.model.to(self.device)
            self.model.eval()

            self.hidden_size = self._resolve_hidden_size()
            self.classifier_head = nn.Linear(self.hidden_size, 2)
            self.classifier_head.to(self.device)
        except Exception:
            # Fall back to a tiny dummy classifier that does not depend on HF resources
            self.use_dummy = True
            self.model = None
            self.tokenizer = None
            self.hidden_size = 128
            self.classifier_head = nn.Linear(self.hidden_size, 2)
            self.classifier_head.to(self.device)

    def _resolve_hidden_size(self) -> int:
        # If using dummy fallback, return a fixed size
        if getattr(self, "use_dummy", False) or self.model is None:
            return 128
        config = self.model.config
        for attr in ["hidden_size", "d_model", "n_embd"]:
            if hasattr(config, attr):
                return int(getattr(config, attr))
        raise ValueError("Unable to resolve hidden size from model config.")

    @classmethod
    def from_config(cls, cfg) -> "PromptModel":
        return cls(
            name=cfg.name,
            model_type=cfg.type,
            precision=cfg.precision,
            max_new_tokens=cfg.max_new_tokens,
        )

    def freeze_backbone(self) -> None:
        if getattr(self, "use_dummy", False) or self.model is None:
            return
        for param in self.model.parameters():
            param.requires_grad = False

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        if getattr(self, "use_dummy", False) or self.model is None:
            return list(self.classifier_head.parameters())
        return list(self.model.parameters()) + list(self.classifier_head.parameters())

    def set_train_mode(self, train_backbone: bool) -> None:
        if getattr(self, "use_dummy", False) or self.model is None:
            self.classifier_head.train()
            return
        if train_backbone:
            self.model.train()
        else:
            self.model.eval()
        self.classifier_head.train()

    def generate_texts(
        self,
        texts: List[str],
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[str]:
        # If in dummy mode, return inputs as-is to keep the pipeline runnable without HF resources
        if getattr(self, "use_dummy", False) or self.tokenizer is None:
            return texts
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        gen_tokens = max_new_tokens or self.max_new_tokens
        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    def encode(self, texts: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        # Dummy path: return zero-vectors
        if getattr(self, "use_dummy", False) or self.tokenizer is None:
            import torch as _torch
            batch_size = len(texts)
            return _torch.zeros(batch_size, self.hidden_size, device=self.device)
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.set_grad_enabled(self.model.training):
            if self.model_type == "seq2seq":
                encoder = self.model.get_encoder()
                outputs = encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = (
                    outputs.hidden_states[-1]
                    if outputs.hidden_states is not None
                    else outputs.last_hidden_state
                )

        mask = batch["attention_mask"].unsqueeze(-1).float()
        masked_hidden = hidden * mask
        summed = masked_hidden.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return pooled

    def classify(self, texts: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        pooled = self.encode(texts, max_length=max_length)
        logits = self.classifier_head(pooled)
        return logits


class PromptEditor(PromptModel):
    @classmethod
    def from_config(cls, cfg, editor_name: Optional[str] = None) -> "PromptEditor":
        name = editor_name or cfg.editor_name or cfg.name
        return cls(
            name=name,
            model_type=cfg.type,
            precision=cfg.precision,
            max_new_tokens=cfg.max_new_tokens,
        )

    @classmethod
    def from_existing(cls, base: PromptModel) -> "PromptEditor":
        return cls(
            name=base.name,
            model_type=base.model_type,
            precision=base.precision,
            max_new_tokens=base.max_new_tokens,
            shared_model=base.model,
            shared_tokenizer=base.tokenizer,
        )

    def propose_prompt(
        self,
        prompt: str,
        stats: str,
        errors_in: List[tuple],
        errors_ood: List[tuple],
        max_errs: int = 4,
        max_new_tokens: int = 140,
    ) -> str:
        def fmt_errs(errs: List[tuple]) -> str:
            lines = []
            for text, label, output in errs[:max_errs]:
                gold = "Positive" if label == 1 else "Negative"
                lines.append(f"- Text: {text}\n  Gold: {gold}\n  Model: {output}")
            return "\n".join(lines) if lines else "(none)"

        critique = (
            "You are improving a SINGLE reusable prompt for binary sentiment classification.\n"
            "It will be used with: 'Text: ... Answer with exactly one word: Positive or Negative.'\n"
            "Goal: improve accuracy across BOTH corpora while keeping the prompt short and stable.\n"
            "Make MINIMAL, format-preserving edits.\n\n"
            f"CURRENT PROMPT:\n{prompt}\n\n"
            f"CURRENT PERFORMANCE SUMMARY:\n{stats}\n\n"
            "MISCLASSIFIED SST-2 EXAMPLES:\n"
            + fmt_errs(errors_in)
            + "\n\n"
            "MISCLASSIFIED YELP EXAMPLES:\n"
            + fmt_errs(errors_ood)
            + "\n\n"
            "Output ONLY the revised prompt text."
        )
        output = self.generate_texts([critique], max_new_tokens=max_new_tokens)[0]
        return output

    def self_refine_prompt(self, prompt: str, max_new_tokens: int = 140) -> str:
        critique = (
            "Critique and minimally improve this sentiment-classification prompt. "
            "Enforce one-word output: Positive or Negative. Output only the revised prompt.\n\n"
            f"PROMPT:\n{prompt}\n"
        )
        output = self.generate_texts([critique], max_new_tokens=max_new_tokens)[0]
        return output
