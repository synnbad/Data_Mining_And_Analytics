import pandas as pd

def main():
    cols = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
    df = pd.read_csv(r'c:\Users\saa24b\Data_Mining_And_Analytics\data\auto-mpg.tab', sep='\t', names=cols, comment='#', na_values='?')
    # Convert horsepower to numeric (some ? values may exist)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

    # Group by cylinders
    grouped = df.groupby('cylinders')
    avg_hp = grouped['horsepower'].mean()
    avg_mpg = grouped['mpg'].mean()

    best_hp_cyl = avg_hp.idxmax()
    best_mpg_cyl = avg_mpg.idxmax()

    print('Average horsepower by cylinders:\n', avg_hp)
    print('\nAverage mpg by cylinders:\n', avg_mpg)
    print('\nAnswer:')
    print('Cars with', int(best_hp_cyl), 'cylinders have the highest average horsepower.')
    print('Cars with', int(best_mpg_cyl), 'cylinders have the highest average mpg.')

if __name__ == '__main__':
    main()
