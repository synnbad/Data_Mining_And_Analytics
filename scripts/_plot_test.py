import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path=r'c:\Users\saa24b\Data_Mining_And_Analytics\data\auto-mpg.tab'
df=pd.read_csv(path, sep='\t', header=0, skiprows=[1], na_values='?')
print('shape', df.shape)
print('columns', df.columns.tolist())
fig,ax=plt.subplots()
df['mpg'].dropna().hist(ax=ax)
out=r'c:\Users\saa24b\Data_Mining_And_Analytics\results\test_hist.png'
fig.savefig(out)
print('wrote', out)
