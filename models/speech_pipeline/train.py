import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Dataset Configuration
DATASET_PATH = "/Users/rohansidharthsamala/Documents/IIITH_three_models/TESS Toronto emotional speech set data"

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
np.save("speech_classes.npy", encoder.classes_) # Save classes for test script

# 2. Train / Val / Test Splits
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=42)

# Save test split to disk so test.py evaluates on the exact same unseen data
test_df.to_csv("speech_test_split.csv", index=False)

# 3. Audio Processing & Dataset
MAX_LEN = 120

def extract_mfcc(path, n_mfcc=40):
    audio, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - len(mfcc)), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]
    return mfcc

class SpeechDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mfcc = extract_mfcc(row["audio_path"])
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(row["label"], dtype=torch.long)

train_loader = DataLoader(SpeechDataset(train_df), batch_size=32, shuffle=True)
val_loader = DataLoader(SpeechDataset(val_df), batch_size=32)

# 4. Model Architecture
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
        return self.fc(output[:, -1, :])

# 5. Training execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechEmotionModel(num_classes=len(encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training Speech Model...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "speech_emotion_model.pth")
print("Speech weights saved successfully!")