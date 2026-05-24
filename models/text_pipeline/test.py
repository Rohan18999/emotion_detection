import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train import TextEmotionModel, TextDataset # Reuses dataset structures natively

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv("text_test_split.csv")
classes = np.load("text_classes.npy", allow_pickle=True)
test_loader = DataLoader(TextDataset(test_df), batch_size=16)

model = TextEmotionModel(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("text_emotion_model.pth", map_location=device))
model.eval()

all_preds, all_labels = [], []
print("Evaluating Text Model on Test Set...")
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        preds = torch.argmax(model(input_ids, attention_mask), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n--- Text Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=classes))

os.makedirs("../../Results/accuracy_tables", exist_ok=True)
os.makedirs("../../Results/plots", exist_ok=True)

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
pd.DataFrame(report).transpose().to_csv("../../Results/accuracy_tables/text_report.csv")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.title("Text Model Confusion Matrix")
plt.savefig("../../Results/plots/text_confusion_matrix.png", bbox_inches='tight')
print("Metrics and plots archived safely inside Results/")