import pandas as pd

def main():
    df = pd.read_csv(r'c:\Users\saa24b\Data_Mining_And_Analytics\data\iris.tab', sep='\t')

    # Mean sepal length for setosa
    setosa = df[df['Species'] == 'setosa']
    mean_sepal_length_setosa = setosa['SepalLength'].mean()

    # Std dev sepal width for virginica
    virginica = df[df['Species'] == 'virginica']
    std_sepal_width_virginica = virginica['SepalWidth'].std()

    # 3rd quartile (75th percentile) of petal length for versicolor
    versicolor = df[df['Species'] == 'versicolor']
    q3_petal_length_versicolor = versicolor['PetalLength'].quantile(0.75)

    # Correlations (Pearson)
    corr_petalwidth_petallength = df['PetalWidth'].corr(df['PetalLength'])
    corr_petalwidth_sepalwidth = df['PetalWidth'].corr(df['SepalWidth'])

    print('Mean sepal length (setosa):', round(mean_sepal_length_setosa, 3))
    print('Std dev sepal width (virginica):', round(std_sepal_width_virginica, 3))
    print('3rd quartile petal length (versicolor):', round(q3_petal_length_versicolor, 3))
    print('Pearson r (petal width, petal length):', round(corr_petalwidth_petallength, 3))
    print('Pearson r (petal width, sepal width):', round(corr_petalwidth_sepalwidth, 3))

    print('\nAnswers:')
    print('Mean sepal length for Iris-setosa =', round(mean_sepal_length_setosa, 3))
    print('Standard deviation of sepal width for Iris-virginica =', round(std_sepal_width_virginica, 3))
    print('Third quartile of petal length for Iris-versicolor =', round(q3_petal_length_versicolor, 3))
    print('Is petal width positively correlated with petal length? ', 'Yes' if corr_petalwidth_petallength > 0 else 'No')
    print('Is petal width positively correlated with sepal width? ', 'Yes' if corr_petalwidth_sepalwidth > 0 else 'No')

if __name__ == '__main__':
    main()
