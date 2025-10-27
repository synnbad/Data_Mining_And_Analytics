import os
import pandas as pd

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path = os.path.join(repo_root, 'data', 'raw', 'auto-mpg.tab')
    # Read file, skip the second row which contains types
    df = pd.read_csv(path, sep='\t', header=0, skiprows=[1], na_values='?')
    # Ensure numeric columns
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')

    grouped = df.groupby('cylinders')
    avg_hp = grouped['horsepower'].mean()
    avg_mpg = grouped['mpg'].mean()

    print('Average horsepower by cylinders:\n', avg_hp)
    print('\nAverage mpg by cylinders:\n', avg_mpg)

    best_hp_cyl = int(avg_hp.idxmax())
    best_mpg_cyl = int(avg_mpg.idxmax())

    print('\nCars with', best_hp_cyl, 'cylinders have the highest average horsepower.')
    print('Cars with', best_mpg_cyl, 'cylinders have the highest average mpg.')

if __name__ == '__main__':
    main()
