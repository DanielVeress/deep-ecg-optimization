import os
import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
SAMPLING_RATE = 100 # 100Hz or 500Hz

# Load annotation data
# See https://www.nature.com/articles/s41597-020-0495-6/tables/3 for columns present.
database_file_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
Y = pd.read_csv(database_file_path, index_col="ecg_id")

# convert label dictionary
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
X = load_raw_data(Y, SAMPLING_RATE, DATA_DIR)

# Load scp_statements.csv for diagnostic aggregation
# See https://www.nature.com/articles/s41597-020-0495-6/tables/13 for columns present.
statements_file_path = os.path.join(DATA_DIR, "scp_statements.csv")
agg_df = pd.read_csv(statements_file_path, index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    """Aggregates and returns a list of superclasses present for each ECG."""
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)



## Info about sets
print(f"Feature data shape (X): {X.shape}") # (Samples, Timepoints, Leads)
print(f"Label data shape (Y): {Y.shape}")

# Flatten the list of diagnostic superclasses to get a total count
all_diagnostics = [item for sublist in Y.diagnostic_superclass for item in sublist]

# Calculate class distribution
class_counts = pd.Series(all_diagnostics).value_counts()
print("\n--- Class Distribution (Diagnostic Superclass) ---")
print(class_counts)

# Check how many samples have multiple superclasses
multi_label_counts = Y.diagnostic_superclass.apply(len).value_counts().sort_index()
print("\n--- Samples by Number of Labels ---")
print(multi_label_counts)

print("\n--- Missing Values in Metadata (Y) ---")
# Check for nulls in the metadata
missing_meta = Y.isnull().sum()
print(missing_meta[missing_meta > 0]) # Only print columns with missing data

print("\n--- Missing/Corrupt Values in Signal Data (X) ---")
# Check if there are any NaNs or Infinite values in the numpy array
has_nans = np.isnan(X).any()
has_infs = np.isinf(X).any()
print(f"X contains NaNs: {has_nans}")
print(f"X contains Infinite values: {has_infs}")

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass