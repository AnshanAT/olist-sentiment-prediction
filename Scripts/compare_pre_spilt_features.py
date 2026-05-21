import pandas as pd
import numpy as np

# Load datasets
notebook = pd.read_csv('./Data/pre_split_features_nb.csv')
script = pd.read_csv('./Data/pre_split_features.csv')

print("Notebook shape:", notebook.shape)
print("Script shape:", script.shape)

# Sort by unique identifier to align rows properly before comparison
if 'order_id' in notebook.columns and 'order_id' in script.columns:
    notebook = notebook.sort_values(by='order_id').reset_index(drop=True)
    script = script.sort_values(by='order_id').reset_index(drop=True)

# Check column alignment
if list(notebook.columns) == list(script.columns):
    print("Column check: PASS")
else:
    print("Column check: FAIL")

# Select numeric columns only
num_cols = notebook.select_dtypes(include=np.number).columns

# Round to ignore minor precision differences from file I/O truncation
notebook_rounded = notebook[num_cols].round(3)
script_rounded = script[num_cols].round(3)

# Compare values
same = notebook_rounded.equals(script_rounded)

if same:
    print("Value check: PASS")
else:
    print("Value check: FAIL")
    diff = notebook_rounded.compare(script_rounded, result_names=("Notebook", "Script"))
    
    print("\nDifferences:")
    print(diff.head(10))

    print("\nError Count per Column:")
    # Extract original column names from the multi-index top level
    original_cols = diff.columns.get_level_values(0).unique()
    
    for col in original_cols:
        # Count rows where Notebook and Script actually differ (not both NaN in diff)
        mismatch_count = diff[col].notna().any(axis=1).sum()
        print(f"Column: {col:<30} | Error Count: {mismatch_count} rows")
        
    print(f"\nTotal Unique Mismatched Rows: {len(diff)} rows")



# ### Review 9088 case in detail
# # Get the exact order_id for row 9088
# idx_9088 = 9088
# order_id_9088 = notebook.loc[idx_9088, 'order_id']
# print("=== Target Order ID ===")
# print(f"Index 9088 Order ID: {order_id_9088}\n")

# # Extract operational date columns for this order
# date_cols = [
#     'order_purchase_timestamp', 
#     'order_delivered_customer_date', 
#     'review_creation_date'
# ]

# print("=== Date Values Comparison ===")
# for col in date_cols:
#     print(f"{col:<32} -> Notebook: {notebook.loc[idx_9088, col]}, Script: {script.loc[idx_9088, col]}")



# ### Check raw review data for this specific order_id for 9088
# try:
#     raw_reviews = pd.read_csv('./Data/olist_order_reviews_dataset.csv')
#     mismatched_review = raw_reviews[raw_reviews['order_id'] == '186c43c492d0813457a0da8ee78ef748']
    
#     print("=== Raw Reviews Table State ===")
#     print(mismatched_review[['order_id', 'review_id', 'review_score', 'review_creation_date']])
# except Exception as e:
#     print(f"Error: {e}")


