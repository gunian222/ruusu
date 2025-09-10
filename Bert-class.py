from datasets import load_dataset  # 导入情感分析数据集
import pandas as pd
from sentry_sdk.utils import epoch

dataset = load_dataset("fancyzhx/ag_news")
print(dataset["train"][0])

from transformers import AutoTokenizer # 分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(example):
    return tokenizer(example["text"],padding="max_length",truncation=True,max_length=512)
tokenized_datasets=dataset.map(tokenize,batched=True)

import torch  # 数据集格式转换
tokenized_datasets = tokenized_datasets.rename_column("label","labels")
tokenized_datasets.set_format("torch",columns=["input_ids", "attention_mask", "labels"])

from torch.utils.data import DataLoader
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  #打乱数据
test_loader = DataLoader(test_dataset, batch_size=64)

# 模型初始化
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           num_labels=4,
                                                           output_attentions=False,
                                                           output_hidden_states=False
                                                           )
total_params= sum(p.numel() for p in model.parameters())
trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数：{total_params/1e6:.1f}M,可训练参数{trainable_params/1e6:.1f}M")


# 训练配置
from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(),lr=2e-5,eps=1e-8)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,num_warmup_steps=0,num_training_steps=total_steps
)

# 训练循环
from tqdm import tqdm
import numpy as np
train_losses = []
val_accuracies = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader,desc= f"Epoch {epoch+1}/{epochs}")

    for batch in progress_bar:
        batch = {k:v.to(device) for k,v in batch.items()}

        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        #反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({"loss":f"{loss.item():.4f}"})

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_correct = 0
    total_samples = 0

    for batch in tqdm(test_loader,desc= "Validating"):
        batch = {k:v.to(device) for k,v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)
    accuracy = total_correct / total_samples
    val_accuracies.append(accuracy)

    print(f"\nEpoch {epoch + 1} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Acc: {accuracy:.4f}\n")

save_dir = "./bert_agnews"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"模型已保存到 {save_dir}")

# 9. 预测示例
def predict(texts):
    model.eval()
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**encodings)
    preds = torch.argmax(outputs.logits, dim=-1)
    return preds.cpu().numpy()


labels = ["World", "Sports", "Business", "Sci/Tech"]
sample_texts = [
    "The stock market crashed yesterday amid fears of inflation.",
    "The football team won the championship after a thrilling match."
]
preds = predict(sample_texts)
for text, label in zip(sample_texts, preds):
    print(f"【{text}】 → 预测类别: {labels[label]}")