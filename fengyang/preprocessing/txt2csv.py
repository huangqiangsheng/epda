import pandas as pd
import numpy as np

data_file_name='data'
txt_file_path='../data/'+data_file_name+'.txt'
labels_cols=[]
for i in [1,2]:
    labels_cols.append('wg_'+str(i)+'_w')
    for j in [1,2,3,4]:
        labels_cols.append('wg_'+str(i)+'_p_'+str(j)+'_x')
        labels_cols.append('wg_' + str(i) + '_p_' + str(j) + '_y')
print(labels_cols)
X_cols_indices=np.linspace(1.45,1.65,51).round(3)
X_cols=[]
for i in range(len(X_cols_indices)):
    X_cols.append('te1_'+str(X_cols_indices[i]))
for i in range(len(X_cols_indices)):
    X_cols.append('te2_'+str(X_cols_indices[i]))
for i in range(len(X_cols_indices)):
    X_cols.append('te_phase_diff_'+str(X_cols_indices[i]))
for i in range(len(X_cols_indices)):
    X_cols.append('tm1_'+str(X_cols_indices[i]))
for i in range(len(X_cols_indices)):
    X_cols.append('tm2_'+str(X_cols_indices[i]))
for i in range(len(X_cols_indices)):
    X_cols.append('tm_phase_diff_'+str(X_cols_indices[i]))

if __name__=='__main__':
    arr=np.loadtxt(txt_file_path)
    print(arr.shape)
    m=arr.shape[0]
    n_features=arr.shape[1]-1
    n_samples=m//52
    labels=np.zeros((n_samples,18))
    X=np.zeros((n_samples,51*6))
    for i in range(n_samples):
        labels[i]=arr[i*52,1:]
        print('labels:',labels[i])
        te1=arr[i*52+1:(i+1)*52,1].reshape(-1,51)
        te2=arr[i*52+1:(i+1)*52,2].reshape(-1,51)
        te_phase_diff=arr[i*52+1:(i+1)*52,3].reshape(-1,51)
        tm1=arr[i*52+1:(i+1)*52,4].reshape(-1,51)
        tm2=arr[i*52+1:(i+1)*52,5].reshape(-1,51)
        tm_phase_diff=arr[i*52+1:(i+1)*52,6].reshape(-1,51)
        X[i]=np.c_[te1,te2,te_phase_diff,tm1,tm2,tm_phase_diff]
    data_csv=pd.DataFrame(data=np.c_[X,labels],columns=X_cols+labels_cols)
    data_csv.to_csv('../data/'+data_file_name+'.csv',index=False)

