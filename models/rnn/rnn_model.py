import torch.nn as nn
from .baseRNN import BaseRNN
import torch
import numpy as np
from torch.utils.data import Dataset
class RNNModel(BaseRNN):
    def __init__(self,input_seq_len,input_dim,hidden_dim,input_dropout_p=0,dropout_p=0,
                 n_layers=1,bidirectional=False,rnn_cell='lstm',output_dim=8):
        super(RNNModel,self).__init__(input_seq_len,hidden_dim,input_dropout_p=input_dropout_p,
                                      dropout_p=dropout_p,n_layers=n_layers,rnn_cell=rnn_cell)
        self.input_dim=input_dim
        self.bidirectional=bidirectional
        self.output_dim=output_dim
        self.linear1=nn.Linear(input_dim,hidden_dim)
        self.relu1=nn.LeakyReLU(0.01)
        self.rnn=self.rnn_cell(hidden_dim,hidden_dim,n_layers,batch_first=True,bidirectional=self.bidirectional,
                               dropout=dropout_p)
        self.linear2=nn.Linear(hidden_dim*input_seq_len,self.output_dim)

    def forward(self,X):
        output=self.relu1(self.linear1(X))
        outputs,hidden=self.rnn(output)
        outputs = self.linear2(outputs.contiguous().view(-1, self.hidden_dim*self.input_seq_len))
        return outputs

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


class EPDARNNDataset(Dataset):
    def __init__(self,X_array,y_array,transform=None):
        super(EPDARNNDataset, self).__init__()
        self.X_array=X_array
        self.y_array=y_array
        self.transform=transform

    def __len__(self):
        return self.X_array.shape[0]

    def __getitem__(self,idx):
        sample={
            'X':self.X_array[idx].reshape(-1,6),
            'y':self.y_array[idx]
        }
        if self.transform is not None:
            sample=self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self,sample):
        X,y=sample['X'],sample['y']
        return {'X':torch.from_numpy(X.astype(np.float32)),
                'y':torch.from_numpy(y.astype(np.float32)),
                }