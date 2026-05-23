import time
import pandas as pd
from cross_val import cross_validate
from models.random_forest import RandomForestModel

start_time = time.time()


df = pd.read_csv("./Data/pre_split_features.csv")
# dataset sampling for faster execution during development
df = df.sample(n=4000,random_state=42).reset_index(drop=True)

param_grid = {
    "max_depth": [5, 10, 20]
}

results = cross_validate(
    model_class=RandomForestModel,
    df=df,
    param_grid=param_grid,
    n_outer=3,
    n_inner=2,
    scoring="f1_macro"
)


pd.DataFrame(results["fold_metrics"]).to_csv("./Results/fold_metrics_rf.csv", index=False)
pd.DataFrame(results["best_params"]).to_csv("./Results/best_params_rf.csv", index=False)

elapsed = time.time() - start_time

print(df["review_score"].value_counts(normalize=True))
print(f"\nDone")
print(f"Execution time: {elapsed:.2f} sec")
