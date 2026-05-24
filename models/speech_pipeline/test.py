import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train import SpeechEmotionModel, SpeechDataset # Reuses code natively from train.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load split configurations and encoder classes saved during training
test_df = pd.read_csv("speech_test_split.csv")
classes = np.load("speech_classes.npy", allow_pickle=True)

test_loader = DataLoader(SpeechDataset(test_df), batch_size=32)

model = SpeechEmotionModel(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("speech_emotion_model.pth", map_location=device))
model.eval()

all_preds, all_labels = [], []
print("Evaluating Speech Model on Test Set...")
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

# Generate and Export Deliverables
print("\n--- Speech Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=classes))

os.makedirs("../../Results/accuracy_tables", exist_ok=True)
os.makedirs("../../Results/plots", exist_ok=True)

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
pd.DataFrame(report).transpose().to_csv("../../Results/accuracy_tables/speech_report.csv")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Oranges")
plt.title("Speech Model Confusion Matrix")
plt.savefig("../../Results/plots/speech_confusion_matrix.png", bbox_inches='tight')
print("Metrics and plots archived safely inside Results/")