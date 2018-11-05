import os
import torch
from util.predictor import Predictor
from util.checkpoint import Checkpoint
raw_input = input  # Python 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device=torch.device('cpu')
if __name__=='__main__':
    checkpoint_path=os.path.join('experiement_default/checkpoints','fc_model')
    checkpoint=Checkpoint.load(checkpoint_path)
    predictor=Predictor(checkpoint.model.to(device))
    data = pd.read_csv('../../data/data.csv')
    input_features=51*6
    X = data.iloc[:, :input_features]
    y_indices = [i for i in range(input_features + 3, input_features + 3 + 4)] + [i for i in range(input_features+12, input_features +16)]
    y = data.iloc[:, y_indices]

    xscaler=StandardScaler()
    yscaler=StandardScaler()
    X=xscaler.fit_transform(X)
    y=yscaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs=torch.from_numpy(np.array(X_test[:5]).astype(np.float32)).to(device)
    preds=predictor.predict(inputs)

    print('target:',yscaler.inverse_transform(np.array(y_test[:5])))
    print('pred:',yscaler.inverse_transform(preds.numpy()))


