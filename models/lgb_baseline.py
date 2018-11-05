import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    data=pd.read_csv('../data/data.csv')
    input_features=51*6
    X=data.iloc[:,:input_features]
    y_indices=[i for i in range(input_features+3,input_features+3+4)]+[i for i in range(input_features+12,input_features+16)]
    y=data.iloc[:,y_indices]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    lgb_multi_model=MultiOutputRegressor(estimator=lgb.LGBMRegressor())
    lgb_multi_model.fit(X_train,y_train)

    y_test_pred=lgb_multi_model.predict(X_test)
    y_train_pred=lgb_multi_model.predict(X_train)
    print('sample_y:',y_test.iloc[0:5])
    print('sample_pred:',y_test_pred[0:5])
    print('train rmse:',np.sqrt(mean_squared_error(y_train,y_train_pred)))
    print('test rmse:',np.sqrt(mean_squared_error(y_test,y_test_pred)))
