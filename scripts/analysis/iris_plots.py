import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(repo_root, 'data', 'raw', 'iris.tab')
    results_dir = os.path.join(repo_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.read_csv(data_path, sep='\t')
    print('Read iris data:', df.shape)
    
    # Basic stats
    print('\nIris Statistics:')
    print(df.describe())
    
    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df, x='SepalLength', y='SepalWidth', hue='Species', ax=ax)
    ax.set_title('Iris Dataset: Sepal Length vs Sepal Width')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'iris_sepal_scatter.png')
    fig.savefig(plot_path)
    plt.close(fig)
    print('Saved plot to', plot_path)

if __name__ == '__main__':
    main()
