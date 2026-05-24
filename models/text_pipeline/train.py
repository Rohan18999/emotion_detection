import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

DATASET_PATH = "/Users/rohansidharthsamala/Documents/IIITH_three_models/TESS Toronto emotional speech set data"

rows = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            parts = file.replace(".wav", "").split("_")
            if len(parts) >= 3:
                rows.append({"transcript": parts[1], "emotion": parts[2]})

df = pd.DataFrame(rows)
df["emotion"] = df["emotion"].str.replace(" (1)", "", regex=False).str.lower().str.strip()

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["emotion"])
np.save("text_classes.npy", encoder.classes_)

clean_df = pd.DataFrame({
    "input_text": "The speaker said the word: " + df["transcript"].str.lower().str.strip(),
    "label": df["label"]
})

train_df, test_df = train_test_split(clean_df, test_size=0.15, stratify=clean_df["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=42)
test_df.to_csv("text_test_split.csv", index=False)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["input_text"].values
        self.labels = dataframe["label"].values
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        encoding = tokenizer(str(self.texts[idx]), padding="max_length", truncation=True, max_length=16, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(TextDataset(train_df), batch_size=16, shuffle=True)

class TextEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(self.dropout(outputs.pooler_output))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextEmotionModel(num_classes=len(encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("Training Text Model...")
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(input_ids, attention_mask), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "text_emotion_model.pth")
print("Text weights saved successfully!")