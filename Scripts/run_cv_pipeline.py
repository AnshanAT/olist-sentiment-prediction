import numpy as np
import pandas as pd

from Scripts.cross_val import cross_validate


# ==========================
# 1. Load pre-split dataset
# ==========================

df = pd.read_csv("./Data/pre_split_features.csv")

target_col = "review_score"

print("Loaded data:", df.shape)


# ==========================
# 2. Choose model
# ==========================

# Temporary test model
# Replace this with real models later
class DummyModel:

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.majority = None

    def fit(self, X, y):
        vals, counts = np.unique(y, return_counts=True)
        self.majority = vals[np.argmax(counts)]

    def predict(self, X):
        return np.full(len(X), self.majority)


param_grid = {
    "threshold": [0.3, 0.5, 0.7]
}


# ==========================
# 3. Run nested CV
# ==========================

results = cross_validate(
    model_class=DummyModel,
    df=df,
    param_grid=param_grid,
    target_col=target_col,
    n_outer=10,
    n_inner=3,
    scoring="f1_macro",
    random_state=42,
    verbose=True
)


# ==========================
# 4. Save results
# ==========================

pd.DataFrame(results["fold_metrics"]).to_csv(
    "./Data/cv_fold_metrics.csv",
    index=False
)

pd.DataFrame(results["best_params"]).to_csv(
    "./Data/cv_best_params.csv",
    index=False
)

pd.Series(results["feature_names"]).to_csv(
    "./Data/cv_feature_names.csv",
    index=False,
    header=False
)

pd.DataFrame({
    "y_true": results["y_true_all"],
    "y_pred": results["y_pred_all"]
}).to_csv(
    "./Data/cv_predictions.csv",
    index=False
)

print("\nSaved CV results.")
print("Mean metrics:", results["mean_metrics"])
print("Std metrics:", results["std_metrics"])