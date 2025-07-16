
# NLP Project: Stance Detection & Topic Modeling

This repository contains two main components:

1. **Stance Detection**: Classifies text into stance labels using a fine-tuned BERT model.
2. **Topic Modeling**: Discovers latent topics from a collection of documents using BERTopic.

---

## 1️⃣ Stance Detection

### 🔍 What It Does

The stance detection model classifies short text (e.g., social media posts) into one of four categories:

- `Pro-Palestine`
- `Pro-Israel`
- `Neutral`
- `Irrelevant`

It uses a pre-trained `bert-base-uncased` model from Hugging Face Transformers, fine-tuned on your provided dataset.

A **hand-annotated dataset** is provided for fine-tuning the model to ensure high-quality training data with accurate stance labels.

The data is split into training, validation, and test sets, tokenized using BERT tokenizer, and passed to a sequence classification head for fine-tuning.

---

### 📁 Input Dataset Format

Your dataset should be a **CSV** file containing at least two columns:

| text                          | label         |
|------------------------------|---------------|
| "We support Palestine."      | Pro-Palestine |
| "I don’t care about politics"| irrelevant     |

Valid `label` values:
- `Pro-Palestine`, `Pro-Israel`, `Neutral`, `irrelevant`

---

### 🖥️ Usage

```bash
python -m stance_model.main <csv_path> --epochs 5
```

### ✅ CLI Arguments

| Argument       | Type | Description                                        |
|----------------|------|----------------------------------------------------|
| `<csv_path>`   | str  | Path to the input CSV file                         |
| `--epochs`     | int  | (Optional) Number of training epochs (default: 3)  |

### 🧾 Example

```bash
python -m stance_model.main dataset/stance_data.csv --epochs 5
```

---

## 2️⃣ Topic Modeling

### 🔍 What It Does

This module performs unsupervised topic modeling using:

- [BERTopic](https://maartengr.github.io/BERTopic/)
- Sentence embeddings from SentenceTransformers (`all-MiniLM-L6-v2`)
- Clustering to discover topics from text

---

### 📁 Input Dataset Format

Your CSV file should contain a text column (e.g., YouTube video titles):

| title                          |
|--------------------------------|
| "Why Gaza is under siege"      |
| "History of Palestine and Israel" |

---

### 🖥️ Usage

```bash
python -m topic_model.main dataset/youtube-videos.csv --print-topics
```

### ✅ CLI Arguments

| Argument          | Type | Description                               |
|-------------------|------|-------------------------------------------|
| `<csv_path>`      | str  | Path to the CSV file                      |
| `--print-topics`  | flag | Print the top words for each discovered topic |

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/karan-0701/SocialMediaAnalysisPipeline.git
cd stance-model
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📤 Output

- **Stance model**: Saved to the `exported_stance_model/` directory.
- **Topic modeling**: Topics and keywords are printed in the console.

---

## 📁 Project Structure

```
stance_topic_project/
├── dataset/                            # CSV files (e.g., stance_data.csv, youtube-videos.csv)
├── exported_stance_model/             # Trained stance model is saved here
├── stance_model/                      # Stance detection module
│   ├── config.py
│   ├── data_utils.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── train.py
│   ├── main.py
│   └── __init__.py
├── topic_model/                       # Topic modeling module
│   ├── data_loader.py
│   ├── model.py
│   ├── utils.py
│   ├── main.py
│   └── __init__.py
├── requirements.txt
├── .gitignore
└── README.md
```

---
