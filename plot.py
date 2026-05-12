import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training accuracy from a CSV file.")
    parser.add_argument("--csv", type=str, default="results/transformer_lm_csv_questions-words_wd1.0_frac0.04_seed42/metrics.csv", help="Path to metrics CSV (default: metrics.csv)")

    # Fix for running in Colab: parse known args and ignore unknown ones
    args, unknown = parser.parse_known_args()

    # Expect columns: step, train_loss, train_acc, test_loss, test_acc
    df = pd.read_csv(args.csv)

    if not {"step", "train_acc", "test_acc"}.issubset(df.columns):
        raise ValueError("CSV must contain at least 'step', 'train_acc', and 'test_acc' columns")

    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["train_acc"], label="Train accuracy")
    plt.plot(df["step"], df["test_acc"], label="Test accuracy") # Added test accuracy plot
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for Transformer architecture on Analogy, WD=1.0, Train fraction=0.04")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("transformer_lm_csv_questions-words_wd1.0_frac0.04.png")


if __name__ == "__main__":
    main()
