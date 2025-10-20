import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(repo_root, 'data', 'diabetes.csv')
results_path = os.path.join(repo_root, 'results', 'weka_lab_results.csv')

# Load data
df = pd.read_csv(data_path)
# map class to 0/1
df['class'] = df['class'].map({'tested_negative':0, 'tested_positive':1})
X = df.drop(columns=['class'])
y = df['class']

# classifiers
classifiers = {
    'NaiveBayes': GaussianNB(),
    'J48': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Logistic': LogisticRegression(max_iter=1000, solver='lbfgs'),
    'SMO': SVC(kernel='rbf', probability=True)
}

# CV setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

variants = {}
# Original
variants['original'] = X.copy()
# Discretized - apply KBinsDiscretizer to all features (uniform bins, 5 bins)
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
variants['discretized'] = pd.DataFrame(kbd.fit_transform(X), columns=X.columns)
# Normalized - MinMaxScaler to 0-1 on first 8 attributes (all features here)
scaler = MinMaxScaler()
variants['normalized'] = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

results = []
for variant_name, Xv in variants.items():
    for clf_name, clf in classifiers.items():
        scores = cross_val_score(clf, Xv, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_acc = scores.mean()
        std_acc = scores.std()
        results.append({'variant': variant_name, 'classifier': clf_name, 'accuracy_mean': mean_acc, 'accuracy_std': std_acc})
        print(f'{variant_name:<12} {clf_name:<12} Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})')

# Save results
pd.DataFrame(results).to_csv(results_path, index=False)
print('\nSaved results to', results_path)
