import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_f1(path):
    df = pd.read_csv(path)
    return df["f1_macro"]


def main():

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )

    results_dir = os.path.join(project_root, "Results")

    files = {
        "LR": {
            "Base": os.path.join(results_dir, "fold_metrics_lr.csv"),
            "Behavioural": os.path.join(results_dir, "fold_metrics_lr_enhanced.csv"),
        },
        "RF": {
            "Base": os.path.join(results_dir, "fold_metrics_rf.csv"),
            "Behavioural": os.path.join(results_dir, "fold_metrics_rf_enhanced.csv"),
        },
        "NODE": {
            "Base": os.path.join(results_dir, "fold_metrics_node.csv"),
            "Behavioural": os.path.join(results_dir, "fold_metrics_node_enhanced.csv"),
        },
    }

    models = []

    base_means = []
    base_stds = []

    enhanced_means = []
    enhanced_stds = []

    for model_name, paths in files.items():

        base_f1 = load_f1(paths["Base"])
        enhanced_f1 = load_f1(paths["Behavioural"])

        models.append(model_name)

        base_means.append(base_f1.mean())
        base_stds.append(base_f1.std(ddof=0))

        enhanced_means.append(enhanced_f1.mean())
        enhanced_stds.append(enhanced_f1.std(ddof=0))


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

    plt.figure(figsize=(10,6))

    plt.bar(
        x - width/2,
        base_means,
        width,
        yerr=base_stds,
        capsize=8,
        label="Base features",
        color="firebrick"
    )

    plt.bar(
        x + width/2,
        enhanced_means,
        width,
        yerr=enhanced_stds,
        capsize=8,
        label="Behavioural features",
        color="royalblue"
    )

    plt.xticks(x, models)

    plt.ylabel("Macro-F1")
    plt.xlabel("Model")

    plt.title(
        "Feature Engineering Effect on Prediction Performance",
        pad=15
    )

    plt.ylim(0.25,0.35)

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.4
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        results_dir,
        "feature_comparison_f1.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()