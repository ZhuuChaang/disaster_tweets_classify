# test.py
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import get_dataloaders, get_tokenizer, DisasterTweetsDataset
from transformers import AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# 配置
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
max_length = 48
model_dir = "./model"  # 训练完成后保存的模型目录

# -----------------------------
# 数据
# -----------------------------
test_df = pd.read_csv("data/test.csv")

# 离线加载 tokenizer
tokenizer = get_tokenizer(model_dir)

# DataLoader
test_dataset = DisasterTweetsDataset(
    texts=test_df["text_clean"].tolist(),
    labels=test_df["target"].tolist(),  # 你自己拆分的测试集有标签
    tokenizer=tokenizer,
    max_length=max_length,
    is_custom_tokenizer=False,
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 模型
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
model.to(device)
model.eval()

# -----------------------------
# 推理
# -----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# -----------------------------
# 评估指标
# -----------------------------
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
