import os
from dataclasses import dataclass
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

CACHE_DIR = ".cache/"


@dataclass
class DatasetBundle:
    pool_in: Optional[Dataset]
    pool_ood: Optional[Dataset]
    dev_in: Dataset
    dev_ood: Dataset
    eval_in: Dataset
    eval_ood: Dataset
    text_field_in: str
    text_field_ood: str
    label_field_in: str
    label_field_ood: str


def parse_dataset_name(name: str) -> Tuple[str, Optional[str]]:
    if "/" in name:
        parts = name.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], "/".join(parts[1:])
    return name, None


def load_hf_dataset(name: str) -> DatasetDict:
    base, subset = parse_dataset_name(name)
    if subset is None:
        return load_dataset(base, cache_dir=CACHE_DIR)
    return load_dataset(base, subset, cache_dir=CACHE_DIR)


def resolve_fields(dataset_name: str, default_text: Optional[str] = None) -> Tuple[str, str]:
    mapping = {
        "glue/sst2": ("sentence", "label"),
        "yelp_polarity": ("text", "label"),
    }
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


def select_range(dataset: Dataset, start: int, size: int) -> Dataset:
    if size <= 0:
        raise ValueError("Requested dataset size must be positive.")
    if start + size > len(dataset):
        raise ValueError("Requested range exceeds dataset length.")
    return dataset.select(range(start, start + size))


def create_train_subsets(
    dataset: Dataset,
    seed: int,
    pool_size: Optional[int],
    dev_size: Optional[int],
) -> Tuple[Optional[Dataset], Dataset]:
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
