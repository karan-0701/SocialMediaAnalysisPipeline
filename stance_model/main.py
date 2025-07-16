# stance_model/main.py

import sys
import argparse
from pathlib import Path
from .data_utils import load_and_preprocess_data, split_data
from .train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train stance detection model.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")

    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        print(f"File {csv_path} does not exist.")
        sys.exit(1)

    df = load_and_preprocess_data(csv_path)
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(df)

    print(f"Training model for {args.epochs} epochs...")
    train_metrics, val_metrics, test_metrics = train_model(
        train_texts, val_texts, test_texts,
        train_labels, val_labels, test_labels,
        num_epochs=args.epochs
    )

    print("\nFinal Evaluation:")
    for name, metrics in zip(["Train", "Val", "Test"], [train_metrics, val_metrics, test_metrics]):
        print(f"{name} → Accuracy: {metrics['eval_Accuracy']:.4f}, F1: {metrics['eval_F1']:.4f}")

if __name__ == "__main__":
    main()
