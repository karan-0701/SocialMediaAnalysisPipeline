import sys
from pathlib import Path
from .data_utils import load_and_preprocess_data, split_data
from .train import train_model

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m stance_model.main <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"File {csv_path} does not exist.")
        sys.exit(1)

    df = load_and_preprocess_data(csv_path)
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(df)

    print("Training model...")
    train_metrics, val_metrics, test_metrics = train_model(
        train_texts, val_texts, test_texts,
        train_labels, val_labels, test_labels
    )

    print("\nFinal Evaluation:")
    for name, metrics in zip(["Train", "Val", "Test"], [train_metrics, val_metrics, test_metrics]):
        print(f"{name} -> Accuracy: {metrics['eval_Accuracy']:.4f}, F1: {metrics['eval_F1']:.4f}")

if __name__ == "__main__":
    main()
