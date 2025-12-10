# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# -----------------------------
# Load Tokenizer (offline supported)
# -----------------------------
def get_tokenizer(tokenizer_dir="./tokenizer"):
    return AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True
    )


# -----------------------------
# Dataset Class
# -----------------------------
class DisasterTweetsDataset(Dataset):
    """
    Dataset for Disaster Tweets.
    Supports:
    - HuggingFace AutoTokenizer (static tokenizer)
    - Custom tokenizer built via HuggingFace Tokenizers library
    """

    def __init__(
        self,
        texts,
        labels=None,
        tokenizer=None,
        max_length=128,
        is_custom_tokenizer=False,
    ):
        """
        Args:
            texts (List[str]): Cleaned texts.
            labels (List[int] or None): For test set, labels may be None.
            tokenizer: HF AutoTokenizer or HuggingFace Tokenizers instance.
            max_length (int)
            is_custom_tokenizer (bool)
        """
        if tokenizer is None:
            raise ValueError("tokenizer must be provided.")

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_custom_tokenizer = is_custom_tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        # 1️⃣ HuggingFace static tokenizer
        if not self.is_custom_tokenizer:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors=None,
            )

            input_ids = torch.tensor(encoded["input_ids"])
            attention_mask = torch.tensor(encoded["attention_mask"])

        # 2️⃣ Custom tokenizer (e.g. BPE via Tokenizers library)
        else:
            enc = self.tokenizer.encode(text)

            ids = enc.ids[: self.max_length]

            pad_len = self.max_length - len(ids)
            if pad_len > 0:
                ids += [0] * pad_len

            attention_mask = [1 if x != 0 else 0 for x in ids]

            input_ids = torch.tensor(ids)
            attention_mask = torch.tensor(attention_mask)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add labels if training/validation
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# -----------------------------
# Dataloader Builder
# -----------------------------
def get_dataloaders(
    train_df,
    val_df=None,
    batch_size=16,
    tokenizer=None,
    max_length=128,
    is_custom_tokenizer=False,
):
    """Return train_loader and val_loader"""

    train_dataset = DisasterTweetsDataset(
        texts=train_df["text_clean"].tolist(),
        labels=train_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        is_custom_tokenizer=is_custom_tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = None
    if val_df is not None:
        val_dataset = DisasterTweetsDataset(
            texts=val_df["text_clean"].tolist(),
            labels=val_df["target"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
            is_custom_tokenizer=is_custom_tokenizer,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loader


# -----------------------------
# Example usage (offline load)
# -----------------------------
if __name__ == "__main__":
    import pandas as pd

    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    tokenizer = get_tokenizer("./distilbert")  # offline

    train_loader, val_loader = get_dataloaders(
        train_df,
        val_df,
        batch_size=16,
        tokenizer=tokenizer,
        max_length=128,
        is_custom_tokenizer=False,
    )

    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})
