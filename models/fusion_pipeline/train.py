# train.py
import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

# Configurations & Paths
DATASET_PATH = "/Users/rohansidharthsamala/Documents/IIITH_three_models/TESS Toronto emotional speech set data"
MAX_AUDIO_LEN = 120

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# =========================================================
# DATA PREPARATION & LOADING PIPELINE
# =========================================================
rows = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            parts = file.replace(".wav", "").split("_")
            if len(parts) >= 3:
                rows.append({
                    "audio_path": path, 
                    "transcript": parts[1], 
                    "emotion": parts[2]
                })

df = pd.DataFrame(rows)
df["emotion"] = df["emotion"].str.replace(" (1)", "", regex=False).str.lower().str.strip()

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["emotion"])
classes = encoder.classes_

# Cache target classes so test.py can parse performance metrics natively
np.save("fusion_classes.npy", classes)

def extract_mfcc(path, n_mfcc=40):
    audio, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) < MAX_AUDIO_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_AUDIO_LEN - len(mfcc)), (0, 0)))
    else:
        mfcc = mfcc[:MAX_AUDIO_LEN]
    return mfcc

class MultimodalDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mfcc = extract_mfcc(row["audio_path"])
        
        text = "The speaker said the word: " + str(row["transcript"]).lower().strip()
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
        
        return {
            "mfcc": torch.tensor(mfcc, dtype=torch.float32),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }

# Strategic Split Mappings (Keeps random states locked symmetrically)
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=42)

# Export split tracking file for test script execution
test_df.to_csv("fusion_test_split.csv", index=False)

train_loader = DataLoader(MultimodalDataset(train_df), batch_size=16, shuffle=True)

# =========================================================
# MODEL ARCHITECTURES
# =========================================================
class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv(x))).permute(0, 2, 1)
        output, _ = self.lstm(x)
        features = output[:, -1, :]
        return self.fc(features), features

class TextEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.pooler_output
        return self.fc(self.dropout(features)), features

class LateFusionModel(nn.Module):
    def __init__(self, speech_model, text_model, num_classes):
        super().__init__()
        self.speech_branch = speech_model
        self.text_branch = text_model
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, mfcc, input_ids, attention_mask):
        _, speech_feats = self.speech_branch(mfcc)
        _, text_feats = self.text_branch(input_ids, attention_mask)
        fused_representation = torch.cat((speech_feats, text_feats), dim=1)
        logits = self.classifier(fused_representation)
        return logits, fused_representation

# =========================================================
# INITIALIZATION & WEIGHT INJECTION
# =========================================================
if __name__ == "__main__":
    print(f"Target Execution Device: {device}")
    
    base_speech = SpeechEmotionModel(num_classes=len(classes))
    base_text = TextEmotionModel(num_classes=len(classes))

    # Weight injection step
    base_speech.load_state_dict(torch.load("/Users/rohansidharthsamala/Documents/IIITH_three_models/speech_emotion_model.pth", map_location=device))
    base_text.load_state_dict(torch.load("/Users/rohansidharthsamala/Documents/IIITH_three_models/text_emotion_model.pth", map_location=device))

    # Freezing foundational layers
    for param in base_speech.parameters():
        param.requires_grad = False
    for param in base_text.parameters():
        param.requires_grad = False

    fusion_model = LateFusionModel(base_speech, base_text, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(fusion_model.classifier.parameters(), lr=1e-3)

    print("\n--- Launching Multimodal Late Fusion Training Loop ---")
    for epoch in range(3):
        fusion_model.train()
        total_loss = 0
        for batch in train_loader:
            mfcc = batch["mfcc"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits, _ = fusion_model(mfcc, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d}/03 | Multimodal Loss: {total_loss:.4f}")

    # Export multi-modal weights to disk
    torch.save(fusion_model.state_dict(), "multimodal_fusion_model.pth")
    print("\n[COMPLETE] Master multimodal weights saved successfully!")