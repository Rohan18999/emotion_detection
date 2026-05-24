# test.py
import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

# Import code assets natively from train script 
from train import LateFusionModel, SpeechEmotionModel, TextEmotionModel, MultimodalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f"Evaluating Fusion Pipeline on device: {device}")
    
    # Load test configuration matrices cached during training step
    test_df = pd.read_csv("fusion_test_split.csv")
    classes = np.load("fusion_classes.npy", allow_pickle=True)
    test_loader = DataLoader(MultimodalDataset(test_df), batch_size=16)

    # Reconstruct empty structural networks
    base_speech = SpeechEmotionModel(num_classes=len(classes))
    base_text = TextEmotionModel(num_classes=len(classes))
    fusion_model = LateFusionModel(base_speech, base_text, num_classes=len(classes)).to(device)
    
    # Inject fine-tuned joint master layer checkpoint
    fusion_model.load_state_dict(torch.load("multimodal_fusion_model.pth", map_location=device))
    fusion_model.eval()

    all_preds, all_labels = [], []
    collected_embeddings = []

    print("\nProcessing Unseen Test Split Sequences...")
    with torch.no_grad():
        for batch in test_loader:
            mfcc = batch["mfcc"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits, fused_feats = fusion_model(mfcc, input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            collected_embeddings.extend(fused_feats.cpu().numpy())

    print("\n================ MULTIMODAL CLASSIFICATION REPORT ================")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Export classification report data matrices
    os.makedirs("/Users/rohansidharthsamala/Documents/IIITH_three_models/Results/accuracy_tables", exist_ok=True)
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv("/Users/rohansidharthsamala/Documents/IIITH_three_models/Results/accuracy_tables/fusion_report.csv")
    print("Saved summary table metrics to: Results/accuracy_tables/fusion_report.csv")

    # =========================================================
    # CLUSTER SPACE VISUALIZATION (t-SNE DIMENSION PLOT)
    # =========================================================
    print("\nCompressing 1024-dimensional fused space via t-SNE for plotting...")
    embeddings_matrix = np.array(collected_embeddings)
    labels_array = np.array(all_labels)

    # Using max_iter instead of deprecated n_iter keyword argument
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    compressed_embeddings = tsne.fit_transform(embeddings_matrix)
    named_labels = [classes[idx] for idx in labels_array]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=compressed_embeddings[:, 0], 
        y=compressed_embeddings[:, 1], 
        hue=named_labels, 
        palette="Dark2", 
        style=named_labels,
        s=60, 
        alpha=0.85
    )
    plt.title("Late-Fusion Multimodal Embedding Space (t-SNE Clustering View)", fontsize=14, weight="bold")
    plt.xlabel("t-SNE Component 1", weight="bold")
    plt.ylabel("t-SNE Component 2", weight="bold")
    plt.legend(title="Target Emotions", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)

    os.makedirs("/Users/rohansidharthsamala/Documents/IIITH_three_models/Results/plots", exist_ok=True)
    plt.savefig("/Users/rohansidharthsamala/Documents/IIITH_three_models/Results/plots/fusion_cluster_space.png", bbox_inches='tight')
    plt.tight_layout()
    print("Saved cluster space plots to: Results/plots/fusion_cluster_space.png")
    print("\n[COMPLETE] Evaluation metrics successfully generated!")