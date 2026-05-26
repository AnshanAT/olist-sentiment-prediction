import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    return pd.read_csv(path)


def main():

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )

    results_dir = os.path.join(project_root, "Results")

    files = {
        "LR": os.path.join(results_dir, "fold_metrics_lr_enhanced.csv"),
        "RF": os.path.join(results_dir, "fold_metrics_rf_enhanced.csv"),
        "NODE": os.path.join(results_dir, "fold_metrics_node_enhanced.csv"),
    }

    models = []

    train_means = []
    train_stds = []

    test_means = []
    test_stds = []

    gaps = []

    for model_name, path in files.items():

        df = load_metrics(path)

        models.append(model_name)

        train_means.append(df["train_f1"].mean())
        train_stds.append(df["train_f1"].std(ddof=0))

        test_means.append(df["f1_macro"].mean())
        test_stds.append(df["f1_macro"].std(ddof=0))

        gaps.append(df["train_test_gap"].mean())

    x = np.arange(len(models))
    width = 0.35

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        train_means,
        width,
        yerr=train_stds,
        capsize=8,
        label="Train Macro-F1",
        color="darkorange"
    )

    plt.bar(
        x + width / 2,
        test_means,
        width,
        yerr=test_stds,
        capsize=8,
        label="Test Macro-F1",
        color="seagreen"
    )

    for i, gap in enumerate(gaps):
        y_pos = max(train_means[i] + train_stds[i], test_means[i] + test_stds[i]) + 0.01
        plt.text(
            x[i],
            y_pos,
            f"Gap={gap:.3f}",
            ha="center",
            va="bottom",
            fontsize=16
        )

    plt.xticks(x, models)

    plt.ylabel("Macro-F1")
    plt.xlabel("Model")

    plt.title(
        "Model Complexity and Generalisation Behaviour",
        pad=15
    )

    plt.ylim(0.25, 0.35)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        results_dir,
        "complexity_generalisation_gap.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()