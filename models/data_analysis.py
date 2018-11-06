import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

labels_cols=[]
for i in [1,2]:
    labels_cols.append('wg_'+str(i)+'_w')
    for j in [1,2,3,4]:
        labels_cols.append('wg_'+str(i)+'_p_'+str(j)+'_x')
        labels_cols.append('wg_' + str(i) + '_p_' + str(j) + '_y')
print(labels_cols)

if __name__=='__main__':
    df=pd.read_csv('../data/data.csv')
    print((df['tm2_1.562']).describe())
    sns.distplot(np.log(df['tm2_1.562']), rug=True, hist=True)
    plt.show()
    sns.distplot(np.log(df['te2_1.562']),rug=True,hist=True)
    plt.show()
    sns.distplot(df['te_phase_diff_1.562'],rug=True,hist=True)
    plt.show()
    for col in df.columns:
        if col.startswith('te1_') or col.startswith('te2_') or col.startswith('tm1_') or col.startswith('tm2_'):
            df[col] = np.log(df[col])
            print(df[col].describe())

    corr=df.corr()
    corr=corr.iloc[:,-18:]
    corr.to_csv('corr.csv', index=True)
    describe=df.describe()
    for col in describe.columns:
        describe[col]=np.round(describe[col],4)
    describe.to_csv('describe.csv',index=True)

    for label_col in labels_cols:
        sns.distplot(df[label_col],rug=True,hist=True)
        plt.title(label_col)
        plt.show()




