import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
if __name__=='__main__':
    df=pd.read_csv('../data/data.csv')
    print((df['tm2_1.45']).describe())
    sns.distplot(df['tm2_1.45'], rug=True, hist=True)
    describe=df.describe()
    for col in describe.columns:
        describe[col]=np.round(describe[col],4)
    describe.to_csv('describe.csv',index=True)
    plt.show()
