import os
import pandas as pd


def get_stats(path):

    df = pd.read_csv(path)

    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]

    result = {}

    for metric in metrics:

        mean = df[metric].mean()
        std = df[metric].std(ddof=0)

        result[metric] = f"{mean:.3f} ± {std:.3f}"

    return result


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

results_dir = os.path.join(
    project_root,
    "Results"
)


files = {
    "Logistic Regression":
        "fold_metrics_lr_enhanced.csv",

    "Random Forest":
        "fold_metrics_rf_enhanced.csv",

    "NODE":
        "fold_metrics_node_enhanced.csv"
}

table = []

for model, filename in files.items():

    stats = get_stats(
        os.path.join(
            results_dir,
            filename
        )
    )

    table.append({
        "Model": model,
        "Accuracy": stats["accuracy"],
        "Precision": stats["precision_macro"],
        "Recall": stats["recall_macro"],
        "Macro-F1": stats["f1_macro"]
    })

df = pd.DataFrame(table)

save_path = os.path.join(
    results_dir,
    "performance_summary.csv"
)

df.to_csv(
    save_path,
    index=False
)

print(df)
print(f"\nSaved: {save_path}")