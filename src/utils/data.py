"""
utils/data.py

Data loading utilities:
  - C4 validation (calibration data for salience computation)
  - WikiText-2 (perplexity evaluation)
  - PTB (perplexity evaluation)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TokenizedDataset(Dataset):
    """Tokenized dataset for calibration or evaluation."""

    def __init__(self, input_ids: torch.Tensor, seq_len: int = 512):
        self.seq_len = seq_len
        # Chunk into fixed-length sequences
        n_tokens = input_ids.numel()
        n_chunks = n_tokens // seq_len
        self.data = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


def get_c4_calibration_dataloader(
    tokenizer,
    n_samples: int = 512,
    seq_len: int = 512,
    batch_size: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Load C4 validation split for calibration.
    This is the calibration data used for salience computation.
    """
    from datasets import load_dataset

    logger.info(f"Loading C4 calibration data ({n_samples} samples, seq_len={seq_len})...")

    try:
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception:
        # Fallback: use a small subset
        logger.warning("C4 streaming failed, trying cached version...")
        dataset = load_dataset("allenai/c4", "en", split="validation[:5000]")

    # Collect text samples
    texts = []
    for sample in dataset:
        texts.append(sample["text"])
        if len(texts) >= n_samples * 4:  # oversample for chunking
            break

    # Tokenize
    combined_text = " ".join(texts)
    encoding = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )

    dataset_obj = TokenizedDataset(encoding["input_ids"][0], seq_len=seq_len)

    # Subsample to n_samples
    if len(dataset_obj) > n_samples:
        indices = torch.randperm(len(dataset_obj), generator=torch.Generator().manual_seed(seed))
        indices = indices[:n_samples]
        from torch.utils.data import Subset
        dataset_obj = Subset(dataset_obj, indices.tolist())

    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"C4 calibration: {len(dataset_obj)} samples, "
                f"{len(dataloader)} batches (bs={batch_size})")
    return dataloader


def get_wikitext2_dataloader(
    tokenizer,
    seq_len: int = 512,
    batch_size: int = 4,
    split: str = "test",
) -> DataLoader:
    """Load WikiText-2 for perplexity evaluation."""
    from datasets import load_dataset

    logger.info(f"Loading WikiText-2 {split}...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])

    encoding = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    dataset_obj = TokenizedDataset(encoding["input_ids"][0], seq_len=seq_len)

    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=2)


def get_ptb_dataloader(
    tokenizer,
    seq_len: int = 512,
    batch_size: int = 4,
    split: str = "test",
) -> DataLoader:
    """Load Penn Treebank for perplexity evaluation."""
    from datasets import load_dataset

    logger.info(f"Loading PTB {split}...")
    try:
        dataset = load_dataset("ptb_text_only", "penn_treebank", split=split)
        text = "\n\n".join(dataset["sentence"])
    except Exception:
        logger.warning("PTB load failed, using wikitext-103 as fallback")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = "\n\n".join(dataset["text"])

    encoding = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    dataset_obj = TokenizedDataset(encoding["input_ids"][0], seq_len=seq_len)

    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=2)
