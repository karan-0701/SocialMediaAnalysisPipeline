# Stance Detection Model

This repository trains a **BERT-based stance detection model** to classify text into four categories:
* `Pro-Palestine`
* `Pro-Israel`
* `Neutral`
* `Irrelevant`

It uses **Hugging Face Transformers**, PyTorch, and a custom training pipeline to fine-tune a `bert-base-uncased` model.

## 🚀 How It Works

The model is trained to **detect stance** from short social media texts (e.g., tweets or captions). Each input text is classified into one of the four predefined stance categories.

A **hand-annotated dataset** is provided for fine-tuning the model to ensure high-quality training data with accurate stance labels.

The data is split into training, validation, and test sets, tokenized using BERT tokenizer, and passed to a sequence classification head for fine-tuning.

## 🛠️ Project Structure

```
stance_model/
├── dataset/                    # Your CSV files go here
├── exported_stance_model/      # Trained model is saved here
├── stance_model/               # Core module
│   ├── config.py
│   ├── data_utils.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── train.py
│   └── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧪 Input Dataset Format

Your dataset should be a **CSV** file with at least the following two columns:

| text | label |
|------|-------|
| "We stand with Palestine." | Pro-Palestine |
| "I don't care about politics" | irrelevant |

The `label` column must contain one of: `Pro-Palestine`, `Pro-Israel`, `Neutral`, or `irrelevant`

## 🖥️ Command-Line Usage

Run the training script using the module interface:

```bash
python -m stance_model.main <csv_path> [--epochs N]
```

## 🔧 CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `<csv_path>` | str | Path to the input dataset CSV file (required) |
| `--epochs` | int | Number of training epochs (default: `3`) |

## 🧾 Example

Train for 5 epochs using `dataset/stance_data.csv`:

```bash
python -m stance_model.main dataset/stance_data.csv --epochs 5
```

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

## 📤 Output

After training:
* Evaluation metrics are printed for train/val/test sets.
* The trained model and tokenizer are saved to: `exported_stance_model/`