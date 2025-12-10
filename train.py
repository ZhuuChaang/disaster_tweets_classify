# train.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification,  get_scheduler
from torch.optim import AdamW
from dataset import get_dataloaders, get_tokenizer
from tqdm import tqdm

# -----------------------------
# 配置
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
max_length = 48
num_epochs = 5
learning_rate = 2e-5

# -----------------------------
# 数据
# -----------------------------
import pandas as pd
train_df = pd.read_csv("./data/train.csv")
val_df = pd.read_csv("./data/val.csv")

train_df['text_clean'] = train_df['text_clean'].fillna("")
val_df['text_clean'] = val_df['text_clean'].fillna("")

tokenizer = get_tokenizer("./distilbert")  # 离线加载
train_loader, val_loader = get_dataloaders(
    train_df,
    val_df,
    batch_size=batch_size,
    tokenizer=tokenizer,
    max_length=max_length,
    is_custom_tokenizer=False
)

# -----------------------------
# 模型
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "./distilbert",
    num_labels=2
)
model.to(device)

# -----------------------------
# 优化器 & 学习率调度
# -----------------------------
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# -----------------------------
# 训练循环
# -----------------------------
from torch.nn import CrossEntropyLoss
loss_fn = CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=False)
    total_loss = 0
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1} Loss {loss.item():.4f}")

    print(f"Epoch {epoch+1} average loss: {total_loss / len(train_loader):.4f}")

    # -----------------------------
    # 验证
    # -----------------------------
    if val_loader is not None:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Validation accuracy: {correct / total:.4f}")

output_dir = "./model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

