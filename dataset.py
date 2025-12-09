# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class DisasterTweetsDataset(Dataset):
    """
    Dataset for Disaster Tweets classification.
    Only uses `text_clean` column as input.
    """
    def __init__(self, texts, labels=None, tokenizer_name="distilbert-base-uncased", max_length=128):
        """
        Args:
            texts (List[str]): List of cleaned tweet texts.
            labels (List[int], optional): Corresponding target labels (0 or 1). Default is None for test set.
            tokenizer_name (str): HuggingFace tokenizer name.
            max_length (int): Maximum token length for padding/truncation.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Tokenize all texts upfront
        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item


def get_dataloaders(train_df, val_df=None, batch_size=16, tokenizer_name="distilbert-base-uncased", max_length=128):
    """
    Returns PyTorch DataLoaders for training and validation.
    
    Args:
        train_df (pd.DataFrame): Training dataframe with 'text_clean' and 'target' columns.
        val_df (pd.DataFrame, optional): Validation dataframe.
        batch_size (int): Batch size.
        tokenizer_name (str): HuggingFace tokenizer.
        max_length (int): Max token length.
    
    Returns:
        train_loader, val_loader (DataLoader or None)
    """
    train_texts = train_df['text_clean'].tolist()
    train_labels = train_df['target'].tolist()

    train_dataset = DisasterTweetsDataset(train_texts, train_labels, tokenizer_name, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_df is not None:
        val_texts = val_df['text_clean'].tolist()
        val_labels = val_df['target'].tolist()
        val_dataset = DisasterTweetsDataset(val_texts, val_labels, tokenizer_name, max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



