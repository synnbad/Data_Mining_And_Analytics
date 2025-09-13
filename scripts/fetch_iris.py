import os
import pandas as pd

def main():
    # Load canonical iris dataset via sklearn (always available with scikit-learn)
    from sklearn.datasets import load_iris
    data = load_iris()
    # sklearn feature names are like 'sepal length (cm)'; normalize to requested names
    df = pd.DataFrame(data.data, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    df['Species'] = pd.Categorical.from_codes(data.target, data.target_names)

    # Save to repo data folder (absolute path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_path = os.path.join(repo_root, 'data', 'iris.tab')
    df.to_csv(out_path, sep='\t', index=False)
    print(f'Saved {df.shape[0]} rows to {out_path}')

if __name__ == '__main__':
    main()
