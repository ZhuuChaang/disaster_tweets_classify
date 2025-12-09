import pandas as pd
import re
from sklearn.model_selection import train_test_split

# -----------------------------
# 读取原始 CSV
# -----------------------------
df = pd.read_csv("./data/tweets.csv")  # 根据你的文件名修改

# -----------------------------
# 文本清理函数
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)        # 去掉 URL
    text = re.sub(r"@\w+", "", text)           # 去掉 @user
    text = re.sub(r"[^a-z0-9\s]", "", text)   # 去掉非字母数字字符
    text = text.strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)

# -----------------------------
# keyword & location 缺失处理
# -----------------------------
df['keyword'] = df['keyword'].fillna("unknown")
df['location'] = df['location'].fillna("unknown")

# -----------------------------
# 只保留处理后的列
# -----------------------------
df = df[['text_clean', 'keyword', 'location', 'target']]

# -----------------------------
# 划分训练/验证/测试集
# -----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['target']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['target']
)

# -----------------------------
# 保存为 CSV
# -----------------------------
train_df.to_csv("./data/train.csv", index=False)
val_df.to_csv("./data/val.csv", index=False)
test_df.to_csv("./data/test.csv", index=False)

print(f"训练集样本数: {len(train_df)}")
print(f"验证集样本数: {len(val_df)}")
print(f"测试集样本数: {len(test_df)}")
