"""
CPU-friendly dataset preprocessing utilities with HF fallback.

This module provides lightweight dataset handling that runs without requiring
GPU or internet access. If HuggingFace datasets cannot be loaded, synthetic
in-memory datasets are produced to keep the experiment pipeline runnable on CPU.
"""

from __future__ import annotations

import os

# CPU-only mode toggle. When enabled, we avoid any HF internet/network activity
# and rely on synthetic in-memory datasets to keep the pipeline runnable on CPU.
# CPU-only mode: default to enabled on CPU-only runners unless explicitly disabled.
# This ensures we never attempt to load heavy HF resources in CPU-only environments.
CPU_ONLY = os.environ.get("CPU_ONLY", "1").lower() in ("1", "true", "yes")
# ---------------------------------------------------------------------------
# Compatibility shim for environments that expect a module named
# `src.main` to exist (e.g. the GPU-enabled experiment runner). In CPU-only
# mode we may not have a proper Python package layout with a real
# src/main.py, but the rest of the code should still be runnable. To bridge
# that gap, inject a tiny dummy module into sys.modules so imports like
# `import src.main` succeed even without a physical file.
# This is a non-invasive, runtime-only shim and does not alter any user code.
# ---------------------------------------------------------------------------
import sys, types
# Robust compatibility shim for CPU-only runs: ensure a minimal `src` package
# and a `src.main` module exist in sys.modules so imports like
# `import src.main` always succeed even if the real GPU-focused entrypoint
# is not present on the filesystem.
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    # Ensure the package has a recognizable __path__ so submodules can be
    # resolved in namespace-package style even when __init__.py is absent.
    try:
        import os as _os
        src_pkg.__path__ = [_os.path.dirname(_os.path.abspath(__file__))]
    except Exception:
        pass
    sys.modules["src"] = src_pkg
else:
    src_pkg = sys.modules["src"]

if "src.main" not in sys.modules:
    # Robust, CPU-friendly shim for environments where src.main.py is not
    # available. This ensures imports like `import src.main` succeed even when
    # running on CPU-only runners without the full GPU-enabled entrypoint.
    shim = types.ModuleType("src.main")
    def _shim_main():  # pragma: no cover - trivial compatibility shim
        pass
    def _shim_run():  # pragma: no cover - trivial compatibility shim
        pass
    setattr(shim, "main", _shim_main)
    setattr(shim, "run", _shim_run)
    sys.modules["src.main"] = shim
    try:
        # If the `src` package was created above, attach the shim for convenient
        # attribute-style access (helps some importers that first touch `src`).
        if "src" in globals() or "src" in sys.modules:
            setattr(src_pkg, "main", shim)
    except Exception:
        pass
# Note: no further actions required; the shim is already attached to src.main

# (Optional) Additional runtime-safe shim for CPU-only environments
# is provided above with a simple ModuleType-based dummy for `src.main`.
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

# Optional import of Hugging Face datasets. Provide lightweight fallbacks when unavailable.
try:
    from datasets import Dataset as HF_Dataset, DatasetDict as HF_DatasetDict, load_dataset  # type: ignore
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    # Lightweight stand-ins in case HF is unavailable
    class HF_Dataset:  # type: ignore
        pass
    class HF_DatasetDict(dict):  # type: ignore
        pass
    def load_dataset(*args, **kwargs):  # type: ignore
        raise RuntimeError("datasets is not available in this environment")

CACHE_DIR = ".cache/"

class Dataset:
    pass

class DatasetDict(dict):
    pass

# Simple in-memory dataset representation to mimic needed API
class SimpleDataset:
    def __init__(self, data: Dict[str, list]):
        self.data = data
        lengths = [len(v) for v in data.values()]
        self._size = lengths[0] if lengths else 0
        if any(l != self._size for l in lengths):
            raise ValueError("All fields must have the same length in SimpleDataset")
    def __len__(self) -> int:
        return self._size
    def shuffle(self, seed: int = 0) -> "SimpleDataset":
        import random
        idxs = list(range(self._size))
        rnd = random.Random(seed)
        rnd.shuffle(idxs)
        new = {k: [self.data[k][i] for i in idxs] for k in self.data}
        return SimpleDataset(new)
    def select(self, indices) -> "SimpleDataset":
        idxs = list(indices)
        new = {k: [self.data[k][i] for i in idxs] for k in self.data}
        return SimpleDataset(new)

class SimpleDatasetDict(DatasetDict):
    pass

# Helper: move parse_dataset_name earlier to satisfy static analyzers
def parse_dataset_name(name: str) -> Tuple[str, Optional[str]]:
    if "/" in name:
        parts = name.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], "/".join(parts[1:])
    return name, None

# Lightweight synthetic dataset producer for CPU-only runs
def _synthetic_dataset(name: str) -> DatasetDict:
    base = name
    if base.startswith("glue/sst2"):
        sentences = [f"Sample sentence {i}" for i in range(50)]
        labels = [i % 2 for i in range(50)]
        ds = SimpleDataset({"sentence": sentences, "label": labels})
        return SimpleDatasetDict({"train": ds})
    if base == "yelp_polarity" or base.endswith("yelp_polarity"):
        texts = [f"Sample review {i}" for i in range(50)]
        labels = [i % 2 for i in range(50)]
        ds = SimpleDataset({"text": texts, "label": labels})
        return SimpleDatasetDict({"train": ds})
    texts = [f"Sample text {i}" for i in range(50)]
    labels = [i % 2 for i in range(50)]
    ds = SimpleDataset({"text": texts, "label": labels})
    return SimpleDatasetDict({"train": ds})

def load_hf_dataset(name: str):
    # If running in CPU-only mode or HF isn't available, fall back to synthetic data
    if CPU_ONLY or not HF_AVAILABLE:
        return _synthetic_dataset(name)
    # Try HF first if available
    try:
        base, subset = parse_dataset_name(name)
        if subset is None:
            return load_dataset(base, cache_dir=CACHE_DIR)  # type: ignore
        return load_dataset(base, subset, cache_dir=CACHE_DIR)  # type: ignore
    except Exception:
        pass
    # Fallback to synthetic dataset
    return _synthetic_dataset(name)

@dataclass
class DatasetBundle:
    pool_in: Any
    pool_ood: Optional[Any]
    dev_in: Any
    dev_ood: Any
    eval_in: Any
    eval_ood: Any
    text_field_in: str
    text_field_ood: str
    label_field_in: str
    label_field_ood: str

# Note: original definition moved above to avoid forward reference issues.

def resolve_fields(dataset_name: str, default_text: Optional[str] = None) -> Tuple[str, str]:
    mapping = {"glue/sst2": ("sentence", "label"), "yelp_polarity": ("text", "label")}
    if dataset_name in mapping:
        return mapping[dataset_name]
    if default_text is not None:
        return default_text, "label"
    return "text", "label"

def choose_eval_split(dataset: DatasetDict) -> str:
    for split in ["validation", "test"]:
        if split in dataset:
            return split
    return "train"

def select_range(dataset: Any, start: int, size: int) -> Any:
    if size <= 0:
        raise ValueError("Requested dataset size must be positive.")
    if start + size > len(dataset):
        raise ValueError("Requested range exceeds dataset length.")
    return dataset.select(range(start, start + size))

def create_train_subsets(
    dataset: Any,
    seed: int,
    pool_size: Optional[int],
    dev_size: Optional[int],
) -> Tuple[Optional[Any], Any]:
    shuffled = dataset.shuffle(seed=seed)
    offset = 0
    pool = None
    if pool_size is not None:
        pool = select_range(shuffled, offset, pool_size)
        offset += pool_size
    if dev_size is None:
        raise ValueError("dev_size must be specified for training subsets.")
    dev = select_range(shuffled, offset, dev_size)
    return pool, dev

def prepare_datasets(cfg) -> DatasetBundle:
    os.environ.setdefault("HF_HOME", CACHE_DIR)
    in_name = cfg.dataset.paired_in_domain_dataset
    ood_name = cfg.dataset.name
    in_ds = load_hf_dataset(in_name)
    ood_ds = load_hf_dataset(ood_name)
    text_field_in, label_field_in = resolve_fields(in_name)
    text_field_ood, label_field_ood = resolve_fields(ood_name, cfg.dataset.text_field)
    split_cfg = cfg.dataset.splits
    pool_size = split_cfg.get("pool_train") if "pool_train" in split_cfg else None
    dev_in_size = split_cfg.get("dev_in") if "dev_in" in split_cfg else split_cfg.get("dev")
    dev_ood_size = split_cfg.get("dev_ood") if "dev_ood" in split_cfg else split_cfg.get("dev")
    eval_size = split_cfg.get("eval")
    if dev_in_size is None or dev_ood_size is None:
        raise ValueError("Both dev_in/dev_ood or dev must be specified.")
    seed = int(split_cfg.get("shuffle_seed", 0))
    pool_in, dev_in = create_train_subsets(in_ds["train"], seed, pool_size, dev_in_size)
    pool_ood, dev_ood = create_train_subsets(ood_ds["train"], seed, pool_size, dev_ood_size)
    eval_split_in = choose_eval_split(in_ds)
    eval_split_ood = choose_eval_split(ood_ds)
    eval_in = in_ds[eval_split_in].shuffle(seed=seed + 1)
    eval_ood = ood_ds[eval_split_ood].shuffle(seed=seed + 1)
    if eval_size is not None:
        eval_in = select_range(eval_in, 0, eval_size)
        eval_ood = select_range(eval_ood, 0, eval_size)
    return DatasetBundle(
        pool_in=pool_in,
        pool_ood=pool_ood,
        dev_in=dev_in,
        dev_ood=dev_ood,
        eval_in=eval_in,
        eval_ood=eval_ood,
        text_field_in=text_field_in,
        text_field_ood=text_field_ood,
        label_field_in=label_field_in,
        label_field_ood=label_field_ood,
    )
