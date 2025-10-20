import pandas as pd
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
excel_path = os.path.join(repo_root, 'data', 'Week7_Activity_diabetes_dataset.xlsx')
results_path = os.path.join(repo_root, 'results', 'weka_lab_results.csv')
out_path = os.path.join(repo_root, 'results', 'accuracy_comparison.csv')

print('Reading Excel:', excel_path)
try:
    xls = pd.ExcelFile(excel_path)
    print('Sheets:', xls.sheet_names)
    df_x = pd.read_excel(xls, sheet_name=0)
    print('Excel head:')
    print(df_x.head())
except Exception as e:
    print('Error reading Excel:', e)
    raise

print('\nReading our results CSV:', results_path)
df_r = pd.read_csv(results_path)
print(df_r)

# Normalize names to match maybe different casing/spaces
# Expect Excel to have columns like Variant, Classifier, Accuracy
cols = df_x.columns.str.lower()
print('\nExcel columns (lower):', list(cols))

# Try to find classifier and accuracy columns
possible_classifier_cols = [c for c in df_x.columns if 'classif' in c.lower() or 'algorithm' in c.lower() or 'model' in c.lower()]
possible_variant_cols = [c for c in df_x.columns if 'variant' in c.lower() or 'dataset' in c.lower()]
possible_accuracy_cols = [c for c in df_x.columns if 'accuracy' in c.lower() or 'correct' in c.lower() or 'percent' in c.lower()]
print('Detected classifier cols:', possible_classifier_cols)
print('Detected variant cols:', possible_variant_cols)
print('Detected accuracy cols:', possible_accuracy_cols)

# Attempt flexible parsing: melt excel to long format if it's a table
# If Excel has classifiers as rows and variants as columns, we'll handle both shapes.

df_excel_long = None
try:
    # If Excel has columns Variant/Classifier/Accuracy directly
    if {'variant','classifier','accuracy'}.issubset(set(c.lower() for c in df_x.columns)):
        df_excel_long = df_x.rename(columns=lambda s: s.strip()).copy()
        df_excel_long = df_excel_long[['variant','classifier','accuracy']]
    else:
        # Try to melt: assume first col is classifier, other cols are variants
        first_col = df_x.columns[0]
        other_cols = list(df_x.columns[1:])
        df_excel_long = df_x.melt(id_vars=[first_col], value_vars=other_cols, var_name='variant', value_name='accuracy')
        df_excel_long = df_excel_long.rename(columns={first_col: 'classifier'})
    # Clean accuracy values (remove % if present)
    df_excel_long['accuracy'] = df_excel_long['accuracy'].astype(str).str.replace('%','').str.strip()
    df_excel_long['accuracy'] = pd.to_numeric(df_excel_long['accuracy'], errors='coerce')
    # If values look like percentages >1, convert to fraction
    if df_excel_long['accuracy'].max() > 1.0:
        df_excel_long['accuracy'] = df_excel_long['accuracy'] / 100.0
except Exception as e:
    print('Failed to transform Excel to long format:', e)
    raise

print('\nParsed Excel long format (first rows):')
print(df_excel_long.head())

# Prepare our results for merge: standardize names
map_classifier = {
    'NaiveBayes': 'NaiveBayes',
    'J48': 'J48',
    'RandomForest': 'RandomForest',
    'Logistic': 'Logistic',
    'SMO': 'SMO'
}
# Ensure variants naming
# Our variants are original, discretized, normalized

df_r2 = df_r.copy()
# Round accuracy

df_r2['accuracy'] = df_r2['accuracy_mean']

# Try to align classifier names in excel to ours by stripping spaces/case
df_excel_long['classifier_norm'] = df_excel_long['classifier'].str.replace('\n',' ').str.strip().str.lower()
df_r2['classifier_norm'] = df_r2['classifier'].str.strip().str.lower()

df_merge = pd.merge(df_excel_long, df_r2, left_on=['classifier_norm','variant'], right_on=['classifier_norm','variant'], how='outer', suffixes=('_excel','_ours'))

# Compute difference
if 'accuracy' in df_merge.columns and 'accuracy_mean' in df_merge.columns:
    df_merge['diff'] = df_merge['accuracy_mean'] - df_merge['accuracy']

print('\nComparison (first rows):')
print(df_merge.head(20))

# Save
try:
    df_merge.to_csv(out_path, index=False)
    print('\nSaved comparison to', out_path)
except Exception as e:
    print('Failed to save comparison CSV:', e)

# Summarize mismatches
if 'diff' in df_merge.columns:
    mismatches = df_merge[ df_merge['diff'].abs() > 0.01 ]
    print(f"\nNumber of cells with >1% difference: {len(mismatches)}")
    if len(mismatches) > 0:
        print(mismatches[['classifier','variant','accuracy','accuracy_mean','diff']])

print('\nDone')
