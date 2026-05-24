# Multimodal Emotion Recognition System (MERS)

A deep learning framework for classifying human emotional states from the Toronto Emotional Speech Set (TESS). The project evaluates emotion recognition across three experimental conditions:

- Speech-only temporal modeling
- Text-only contextual modeling
- Multimodal late fusion of speech and text features

---

## Student Information

| Field | Details |
| :--- | :--- |
| Name | Samala Rohan Sidharth |
| Roll Number | 23E51A66A1 |
| Email | 23e51a66a1@hitam.org |
| Institution | Hyderabad Institute of Technology and Management (HITAM) |
| Department | Computer Science and Engineering - Artificial Intelligence & Machine Learning |

---

## Project Architecture

The system is divided into three independent pipelines inside the `models/` directory.

| Pipeline | Location | Description |
| :--- | :--- | :--- |
| Speech Pipeline | `models/speech_pipeline/` | Uses MFCC audio features with a Conv1D + BiLSTM model for acoustic emotion classification. |
| Text Pipeline | `models/text_pipeline/` | Uses `bert-base-uncased` to extract semantic features from transcript text. |
| Fusion Pipeline | `models/fusion_pipeline/` | Combines speech and text embeddings into a 1024-dimensional representation and trains a late-fusion classifier. |

---

## Experimental Results

| Modality Pipeline | Test Accuracy | Evaluation Insight |
| :--- | :---: | :--- |
| Text-only | ~14.28% | Performs close to random chance because TESS uses repeated phrases, so the text contains little emotional variation. |
| Speech-only | ~99.00% | Performs strongly because pitch, intensity, and acoustic patterns provide clear emotional cues. |
| Multimodal late fusion | ~99.00% | Learns to rely mainly on useful acoustic embeddings while reducing the impact of weak text features. |

---

## Repository Structure

```text
IIITH_three_models/
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   ├── text_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   └── fusion_pipeline/
│       ├── train.py
│       └── test.py
├── Results/
│   ├── accuracy_tables/                       # Generated CSV reports for each pipeline
│   └── plots/                                 # Generated plots for each pipeline
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone or open the project

Open a terminal in the project root directory:

```bash
cd /Users/rohansidharthsamala/Documents/IIITH_three_models
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Execution

Run the pipelines in the order below. The fusion pipeline depends on the trained speech and text checkpoints.

### Step 1: Train and evaluate the speech model

```bash
cd models/speech_pipeline
python train.py
python test.py
```

Expected generated files:

```text
models/speech_pipeline/speech_emotion_model.pth
models/speech_pipeline/speech_classes.npy
models/speech_pipeline/speech_test_split.csv
Results/accuracy_tables/speech_report.csv
Results/plots/speech_confusion_matrix.png
```

### Step 2: Train and evaluate the text model

```bash
cd ../text_pipeline
python train.py
python test.py
```

Expected generated files:

```text
models/text_pipeline/text_emotion_model.pth
models/text_pipeline/text_classes.npy
models/text_pipeline/text_test_split.csv
Results/accuracy_tables/text_report.csv
Results/plots/text_confusion_matrix.png
```

### Step 3: Prepare checkpoints for fusion training

The fusion training script loads the speech and text checkpoints from the project root. After retraining the individual models, copy the latest checkpoints to the root directory:

```bash
cd ../..
cp models/speech_pipeline/speech_emotion_model.pth speech_emotion_model.pth
cp models/text_pipeline/text_emotion_model.pth text_emotion_model.pth
```

### Step 4: Train and evaluate the multimodal fusion model

```bash
cd models/fusion_pipeline
python train.py
python test.py
```

Expected generated files:

```text
models/fusion_pipeline/multimodal_fusion_model.pth
models/fusion_pipeline/fusion_classes.npy
models/fusion_pipeline/fusion_test_split.csv
Results/accuracy_tables/fusion_report.csv
Results/plots/fusion_cluster_space.png
```

---

## Output Summary

After successful execution, the main deliverables are stored in:

```text
Results/accuracy_tables/
Results/plots/
```

Checkpoint files are saved either inside the individual pipeline folders or in the project root, depending on how the scripts are executed.

---

## Troubleshooting

- If `ModuleNotFoundError` appears, reactivate the virtual environment and rerun `pip install -r requirements.txt`.
- If the dataset is not found, confirm that the TESS folder name and `DATASET_PATH` match exactly.
- If fusion training fails while loading checkpoints, confirm that `speech_emotion_model.pth` and `text_emotion_model.pth` exist in the project root.
- If BERT files fail to download, check the internet connection and rerun the text or fusion pipeline.

---

## Requirements

The project uses the following major libraries:

- PyTorch
- Transformers
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
