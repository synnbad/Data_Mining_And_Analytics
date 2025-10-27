# Data Mining and Analytics

This repository contains coursework and experiments for LIS 5765 (Graduate Data Mining and Analytics Course).

## Repository Structure

```
Data_Mining_And_Analytics/
├── data/
│   ├── raw/                    # Original source datasets
│   │   ├── iris.tab            # Iris dataset (150 samples)
│   │   ├── auto-mpg.tab        # Auto MPG dataset
│   │   └── diabetes.arff       # Pima Indians Diabetes (ARFF format)
│   └── processed/              # Converted/processed datasets
│       ├── diabetes.csv        # Diabetes dataset (CSV format)
│       └── Week7_Activity_diabetes_dataset.*
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── setup.ipynb             # Environment setup
│   ├── analysis.ipynb          # Iris analysis
│   ├── auto_mpg_analysis.ipynb # Auto-MPG analysis
│   └── LIS5765_Data_Exploration.ipynb
├── scripts/
│   ├── analysis/               # Dataset-specific analysis scripts
│   │   ├── iris_stats.py       # Iris statistical analysis
│   │   ├── iris_plots.py       # Iris visualization
│   │   ├── auto_mpg_stats.py   # Auto-MPG statistics
│   │   └── auto_mpg_plots.py   # Auto-MPG visualizations
│   ├── experiments/            # Machine learning experiments
│   │   ├── weka_lab.py         # Main Week 7 Weka lab (10-fold CV)
│   │   ├── compare_weka_results.py
│   │   ├── complete_comparison.py
│   │   ├── confirm_randomforest.py
│   │   └── rf_variation_quick.py
│   └── utils/                  # Utility scripts
│       ├── fetch_iris.py       # Download iris dataset
│       └── convert_arff_to_csv.py  # ARFF to CSV converter
├── results/                    # Experiment results and outputs
│   ├── weka_lab_results.csv    # 10-fold CV accuracy results
│   ├── detailed_comparison.csv # Comparison with Weka reference
│   └── *.png                   # Generated visualizations
└── requirements.txt            # Python dependencies
```

## Key Experiments

### Week 7: Weka Lab Replication (Python)
- **Script**: `scripts/experiments/weka_lab.py`
- **Objective**: Replicate Weka lab using scikit-learn with 10-fold cross-validation
- **Classifiers**: NaiveBayes, J48 (DecisionTree), RandomForest, Logistic, SMO (SVC)
- **Datasets**: Original, Discretized (5 bins), Normalized (0-1)
- **Results**: Average 1.81% difference from Weka reference (excellent match!)

### Iris Dataset Analysis
- Statistical analysis of Iris species characteristics
- Mean, std dev, quartile calculations
- Correlation analysis between features

### Auto-MPG Analysis
- Cylinder group comparisons (horsepower and MPG)
- Visualizations of relationships between variables

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/synnbad/Data_Mining_And_Analytics.git
cd Data_Mining_And_Analytics
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the environment:
- **Windows**: `venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Run Weka Lab Experiment
```bash
python scripts/experiments/weka_lab.py
```

### Compare Results with Reference
```bash
python scripts/experiments/complete_comparison.py
```

### Generate Visualizations
```bash
python scripts/analysis/auto_mpg_plots.py
python scripts/analysis/iris_plots.py
```

## Results Summary

All experiment results are saved in the `results/` folder:
- Machine learning accuracy metrics (CSV format)
- Comparison tables with Weka reference values
- Statistical analysis outputs
- Visualization plots (PNG format)

## Dependencies

Key Python packages:
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib/seaborn**: Visualization
- **openpyxl**: Excel file handling

See `requirements.txt` for complete list.

## Course Information

- **Course**: LIS 5765 - Data Mining and Analytics
- **Institution**: Graduate Program
- **Purpose**: Hands-on practice and coursework experimentation

## License

This repository is for educational purposes.
