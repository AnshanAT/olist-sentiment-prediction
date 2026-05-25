import time
import pandas as pd
from Scripts.corss_validation.cross_val import cross_validate
from models.random_forest import RandomForestModel

start_time = time.time()

# load data
df = pd.read_csv("./Data/pre_split_features.csv")
# dataset sampling for faster execution during development
df = df.sample(n=4000,random_state=42).reset_index(drop=True)

param_grid = {
    "max_depth": [5, 10, 20]
}

print("\n=== Base engineered features ===")
results = cross_validate(
    model_class=RandomForestModel,
    df=df,
    param_grid=param_grid,
    n_outer=3,
    n_inner=2,
    scoring="f1_macro",
    use_interactions=False
)

print("\n=== Enhanced behavioural features ===")
results_enhanced = cross_validate(
    model_class=RandomForestModel,
    df=df,
    param_grid=param_grid,
    n_outer=3,
    n_inner=2,
    scoring="f1_macro",
    use_interactions=True
)

# save results
pd.DataFrame(results["fold_metrics"]).to_csv("./Results/fold_metrics_rf.csv", index=False)
pd.DataFrame(results["best_params"]).to_csv("./Results/best_params_rf.csv", index=False)

pd.DataFrame(results_enhanced["fold_metrics"]).to_csv("./Results/fold_metrics_rf_enhanced.csv", index=False)
pd.DataFrame(results_enhanced["best_params"]).to_csv("./Results/best_params_rf_enhanced.csv", index=False)
                                                     
elapsed = time.time() - start_time

print("\nClass distribution:")
print(df["review_score"].value_counts(normalize=True))

print(f"\nDone")
print(f"Execution time: {elapsed:.2f} sec")
