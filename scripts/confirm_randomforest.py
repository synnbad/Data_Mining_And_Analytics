import pandas as pd

# Load data
weka = pd.read_csv('data/Week7_Activity_diabetes_dataset.csv')
python_res = pd.read_csv('results/weka_lab_results.csv')

# Convert to percentage
python_res['accuracy_pct'] = python_res['accuracy_mean'] * 100
python_res['accuracy_std_pct'] = python_res['accuracy_std'] * 100

# Filter for RandomForest
rf_python = python_res[python_res['classifier'] == 'RandomForest'].copy()

print('=' * 70)
print('RANDOM FOREST ACCURACY CONFIRMATION')
print('=' * 70)

print('\n📊 Weka Reference Results (from Week 7 Activity):')
print('-' * 70)
for dataset in ['Original', 'Discretized', 'Normalized']:
    weka_val = weka.loc[weka['Dataset'] == dataset, 'RandomForest'].values[0]
    print(f'  {dataset:12}  {weka_val:6.4f}%')

print('\n📊 Our Python Results (10-fold Cross-Validation):')
print('-' * 70)
for _, row in rf_python.iterrows():
    print(f'  {row["variant"].capitalize():12}  {row["accuracy_pct"]:6.4f}% (±{row["accuracy_std_pct"]:.2f}%)')

print('\n📈 Differences (Python - Weka):')
print('-' * 70)
for _, row in rf_python.iterrows():
    dataset = row['variant'].capitalize()
    weka_val = weka.loc[weka['Dataset'] == dataset, 'RandomForest'].values[0]
    python_val = row['accuracy_pct']
    diff = python_val - weka_val
    
    if abs(diff) < 1:
        status = '✓ EXCELLENT (< 1%)'
    elif abs(diff) < 2:
        status = '✓ GOOD (< 2%)'
    elif abs(diff) < 3:
        status = '✓ ACCEPTABLE (< 3%)'
    else:
        status = '⚠ NOTABLE (>= 3%)'
    
    print(f'  {dataset:12}  {diff:+6.2f}%  {status}')

print('\n' + '=' * 70)
print('✅ CONFIRMATION SUMMARY:')
print('=' * 70)

# Calculate average difference
diffs = []
for _, row in rf_python.iterrows():
    dataset = row['variant'].capitalize()
    weka_val = weka.loc[weka['Dataset'] == dataset, 'RandomForest'].values[0]
    python_val = row['accuracy_pct']
    diffs.append(abs(python_val - weka_val))

avg_diff = sum(diffs) / len(diffs)
max_diff = max(diffs)

print(f'\nAverage Absolute Difference: {avg_diff:.2f}%')
print(f'Maximum Absolute Difference: {max_diff:.2f}%')

if avg_diff < 1.5:
    print('\n✅ RandomForest results are EXCELLENT - very close to Weka!')
elif avg_diff < 2.5:
    print('\n✓ RandomForest results are GOOD - reasonably close to Weka')
else:
    print('\n⚠ RandomForest results show some variation from Weka')

print('\nNote: Small differences are expected due to:')
print('  • Different random seeds in cross-validation')
print('  • Minor implementation differences between Weka and scikit-learn')
print('=' * 70)
