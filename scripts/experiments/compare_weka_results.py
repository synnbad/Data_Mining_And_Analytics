import os
import pandas as pd
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load our Python results
python_results = pd.read_csv(os.path.join(repo_root, 'results', 'weka_lab_results.csv'))

# Load Week 7 reference results
week7_results = pd.read_csv(os.path.join(repo_root, 'data', 'Week7_Activity_diabetes_dataset.csv'))

# Convert our results to percentage and reshape for comparison
python_df = python_results.copy()
python_df['accuracy_pct'] = python_df['accuracy_mean'] * 100  # Convert to percentage

# Reshape Python results to match Week 7 format
python_pivot = python_df.pivot(index='variant', columns='classifier', values='accuracy_pct')
python_pivot['Average'] = python_pivot.mean(axis=1)
python_pivot = python_pivot.reset_index()
python_pivot['variant'] = python_pivot['variant'].str.capitalize()
python_pivot = python_pivot.rename(columns={'variant': 'Dataset'})

# Reorder columns to match Week 7
column_order = ['Dataset', 'NaiveBayes', 'J48', 'RandomForest', 'Logistic', 'SMO', 'Average']
python_pivot = python_pivot[column_order]

print("="*80)
print("WEKA LAB ASSIGNMENT RESULTS COMPARISON")
print("="*80)
print("\n📊 WEEK 7 REFERENCE RESULTS (from Weka):")
print(week7_results.to_string(index=False))

print("\n\n📊 OUR PYTHON RESULTS (10-fold Cross-Validation):")
print(python_pivot.to_string(index=False, float_format='%.4f'))

# Calculate differences
print("\n\n📈 ACCURACY DIFFERENCES (Python - Weka):")
print("="*80)

comparison = []
for idx, row in week7_results.iterrows():
    dataset = row['Dataset']
    python_row = python_pivot[python_pivot['Dataset'] == dataset].iloc[0]
    
    print(f"\n{dataset} Dataset:")
    print("-" * 60)
    
    for classifier in ['NaiveBayes', 'J48', 'RandomForest', 'Logistic', 'SMO']:
        weka_acc = row[classifier]
        python_acc = python_row[classifier]
        diff = python_acc - weka_acc
        
        # Color coding for terminal
        status = "✓" if abs(diff) < 2 else ("⚠" if abs(diff) < 5 else "✗")
        
        print(f"  {classifier:15} | Weka: {weka_acc:6.2f}% | Python: {python_acc:6.2f}% | Diff: {diff:+6.2f}% {status}")
        
        comparison.append({
            'Dataset': dataset,
            'Classifier': classifier,
            'Weka_Accuracy': weka_acc,
            'Python_Accuracy': python_acc,
            'Difference': diff,
            'Abs_Difference': abs(diff)
        })

# Summary statistics
comparison_df = pd.DataFrame(comparison)
print("\n\n📊 SUMMARY STATISTICS:")
print("="*80)
print(f"Mean Absolute Difference: {comparison_df['Abs_Difference'].mean():.2f} percentage points")
print(f"Max Absolute Difference: {comparison_df['Abs_Difference'].max():.2f} percentage points")
print(f"Min Absolute Difference: {comparison_df['Abs_Difference'].min():.2f} percentage points")
print(f"Std Dev of Differences: {comparison_df['Abs_Difference'].std():.2f} percentage points")

# Find largest differences
print("\n\n🔍 LARGEST DIFFERENCES (Top 5):")
print("-" * 60)
top_diffs = comparison_df.nlargest(5, 'Abs_Difference')
for idx, row in top_diffs.iterrows():
    print(f"  {row['Dataset']:12} | {row['Classifier']:15} | Diff: {row['Difference']:+6.2f}%")

# Analysis by classifier
print("\n\n📊 AVERAGE DIFFERENCE BY CLASSIFIER:")
print("-" * 60)
by_classifier = comparison_df.groupby('Classifier')['Abs_Difference'].agg(['mean', 'max'])
for classifier, row in by_classifier.iterrows():
    print(f"  {classifier:15} | Avg: {row['mean']:5.2f}% | Max: {row['max']:5.2f}%")

# Analysis by dataset variant
print("\n\n📊 AVERAGE DIFFERENCE BY DATASET VARIANT:")
print("-" * 60)
by_dataset = comparison_df.groupby('Dataset')['Abs_Difference'].agg(['mean', 'max'])
for dataset, row in by_dataset.iterrows():
    print(f"  {dataset:12} | Avg: {row['mean']:5.2f}% | Max: {row['max']:5.2f}%")

# Save detailed comparison
output_path = os.path.join(repo_root, 'results', 'detailed_comparison.csv')
comparison_df.to_csv(output_path, index=False)
print(f"\n\n💾 Detailed comparison saved to: {output_path}")

# Final verdict
print("\n\n🎯 CONCLUSION:")
print("="*80)
avg_diff = comparison_df['Abs_Difference'].mean()
if avg_diff < 2:
    print("✅ EXCELLENT: Results are very close to Weka (avg diff < 2%)")
elif avg_diff < 3:
    print("✓ GOOD: Results are reasonably close to Weka (avg diff < 3%)")
elif avg_diff < 5:
    print("⚠ ACCEPTABLE: Results show some variation from Weka (avg diff < 5%)")
else:
    print("✗ SIGNIFICANT: Results differ notably from Weka (avg diff >= 5%)")

print("\nNote: Some differences are expected due to:")
print("  • Different random seeds in cross-validation splits")
print("  • Implementation differences between Weka and scikit-learn")
print("  • Different discretization methods (quantile vs. Weka's method)")
print("  • Different hyperparameter defaults (especially for J48/DecisionTree)")
print("="*80)
