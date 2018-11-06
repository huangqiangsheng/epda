import os
import torch
from util.predictor import Predictor
from util.checkpoint import Checkpoint
from sklearn.metrics import mean_squared_error
raw_input = input  # Python 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import copy
device=torch.device('cpu')
if __name__=='__main__':
    checkpoint_path=os.path.join('checkpoints/checkpoints','fc_model')
    checkpoint=Checkpoint.load(checkpoint_path)
    predictor=Predictor(checkpoint.model.to(device))
    data = pd.read_csv('../../data/data.csv')
    input_features=51*6
    X = data.iloc[:, :input_features]
    for col in X.columns:
        if col.startswith('te1_') or col.startswith('te2_') or col.startswith('tm1_') or col.startswith('tm2_'):
            X.loc[:,col] = np.log(copy.deepcopy(X.loc[:,col]))

    y_indices = [i for i in range(input_features + 3, input_features + 3 + 4)] + [i for i in range(input_features+12, input_features +16)]
    y = data.iloc[:, y_indices]

    xscaler=MinMaxScaler()
    yscaler=MinMaxScaler()
    X=xscaler.fit_transform(X)
    y=yscaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs=torch.from_numpy(np.array(X_train[:40]).astype(np.float32)).to(device)
    preds=predictor.predict(inputs)

    targets=yscaler.inverse_transform(np.array(y_train[:40]))
    preds=yscaler.inverse_transform(preds.numpy())
    print('target:',targets)
    print('pred:',preds)
    print('rmse:',np.sqrt(mean_squared_error(targets,preds)))



