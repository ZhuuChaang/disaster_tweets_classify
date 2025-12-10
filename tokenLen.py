import pandas as pd
from tqdm import tqdm
from dataset import get_tokenizer
import numpy as np

def get_length_distribution(df, tokenizer):
    lengths = []
    for text in tqdm(df["text_clean"]):    
        if not isinstance(text, str):
            continue
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))
    return lengths

train_df = pd.read_csv("data/train.csv")

tokenizer = get_tokenizer("./distilbert")
lengths = get_length_distribution(train_df, tokenizer)


print("Max length:", max(lengths))
print("90% percentile:", np.percentile(lengths, 90))
print("95% percentile:", np.percentile(lengths, 95))
print("99% percentile:", np.percentile(lengths, 99))
