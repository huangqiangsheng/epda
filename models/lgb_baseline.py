import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

if __name__=='__main__':
    data=pd.read_csv('../data/data.csv')
    input_features=51*6
    X=data.iloc[:,:input_features]
    y_indices=[i for i in range(input_features+3,input_features+3+4)]+[i for i in range(input_features+12,input_features+16)]
    y=data.iloc[:,y_indices]
    print(X.describe())
    for col in X.columns:
        if col.startswith('te1_') or col.startswith('te2_') or col.startswith('tm1_') or col.startswith('tm2_'):
            X[col] = np.log(X[col])

    xscaler=MinMaxScaler()
    yscaler=MinMaxScaler()
    X=xscaler.fit_transform(X)
    y=yscaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    lgb_multi_model=MultiOutputRegressor(estimator=lgb.LGBMRegressor())
    lgb_multi_model.fit(X_train,y_train)

    y_test_pred=lgb_multi_model.predict(X_test)
    y_train_pred=lgb_multi_model.predict(X_train)
    print('sample_y:',y_test[0:5])
    print('sample_pred:',y_test_pred[0:5])
    print('train rmse:',np.sqrt(mean_squared_error(yscaler.inverse_transform(y_train),yscaler.inverse_transform(y_train_pred))))
    print('test rmse:',np.sqrt(mean_squared_error(yscaler.inverse_transform(y_test),yscaler.inverse_transform(y_test_pred))))
