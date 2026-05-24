import time
import pandas as pd
from cross_val import cross_validate
from models.node import NODEModel

start_time = time.time()

# load data
df = pd.read_csv("./Data/pre_split_features.csv")
# dataset sampling for faster execution during development
df = df.sample(n=4000,random_state=42).reset_index(drop=True)


param_grid = {
    "num_layers": [4],
    "depth": [4, 6],
    "tree_dim": [2]
}

# param_grid = {
#     "num_layers": [2, 4, 8],
#     "depth": [4, 6, 8],
#     "tree_dim": [1, 2, 3],
# }

print("\n=== Base engineered features ===")
results = cross_validate(
    model_class=NODEModel,
    df=df,
    param_grid=param_grid,
    n_outer=3,
    n_inner=2,
    scoring="f1_macro",
    use_interactions=False
)

print("\n=== Enhanced behavioural features ===")
results_enhanced = cross_validate(
    model_class=NODEModel,
    df=df,
    param_grid=param_grid,
    n_outer=3,
    n_inner=2,
    scoring="f1_macro",
    use_interactions=True
)

# save results
pd.DataFrame(results["fold_metrics"]).to_csv("./Results/fold_metrics_node.csv", index=False)
pd.DataFrame(results["best_params"]).to_csv("./Results/best_params_node.csv", index=False)

pd.DataFrame(results_enhanced["fold_metrics"]).to_csv("./Results/fold_metrics_node_enhanced.csv", index=False)
pd.DataFrame(results_enhanced["best_params"]).to_csv("./Results/best_params_node_enhanced.csv", index=False)

elapsed = time.time() - start_time

print("\nClass distribution:")
print(df["review_score"].value_counts(normalize=True))

print("\nDone")
print(f"Execution time: {elapsed:.2f} sec")