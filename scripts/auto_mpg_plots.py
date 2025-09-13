import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(repo_root, 'data', 'auto-mpg.tab')
    results_dir = os.path.join(repo_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(data_path, sep='\t', header=0, skiprows=[1], na_values='?')
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')
    print('Read df shape:', df.shape)
    print('Results dir:', results_dir)

    grouped = df.groupby('cylinders')
    avg_hp = grouped['horsepower'].mean()
    avg_mpg = grouped['mpg'].mean()

    # Bar plot: average horsepower by cylinders
    fig, ax = plt.subplots(figsize=(6,4))
    avg_hp.plot(kind='bar', color='C1', ax=ax)
    ax.set_xlabel('Cylinders')
    ax.set_ylabel('Average Horsepower')
    ax.set_title('Average Horsepower by Cylinder Count')
    plt.tight_layout()
    hp_path = os.path.join(results_dir, 'avg_horsepower_by_cylinders.png')
    fig.savefig(hp_path)
    plt.close(fig)
    print('Saved hp plot to', hp_path)

    # Bar plot: average mpg by cylinders
    fig, ax = plt.subplots(figsize=(6,4))
    avg_mpg.plot(kind='bar', color='C2', ax=ax)
    ax.set_xlabel('Cylinders')
    ax.set_ylabel('Average MPG')
    ax.set_title('Average MPG by Cylinder Count')
    plt.tight_layout()
    mpg_path = os.path.join(results_dir, 'avg_mpg_by_cylinders.png')
    fig.savefig(mpg_path)
    plt.close(fig)
    print('Saved mpg plot to', mpg_path)

    # Scatter: mpg vs horsepower with cylinder hue
    fig, ax = plt.subplots(figsize=(6,5))
    sns.scatterplot(data=df, x='horsepower', y='mpg', hue='cylinders', palette='tab10', ax=ax)
    ax.set_title('MPG vs Horsepower (colored by cylinders)')
    plt.tight_layout()
    scatter_path = os.path.join(results_dir, 'mpg_vs_horsepower.png')
    fig.savefig(scatter_path)
    plt.close(fig)
    print('Saved scatter plot to', scatter_path)

    print('Saved plots:')
    print(hp_path)
    print(mpg_path)
    print(scatter_path)

if __name__ == '__main__':
    main()
