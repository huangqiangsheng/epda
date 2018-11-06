import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np

class FCNet(nn.Module):
    def __init__(self,input_features,output_features):
        super(FCNet,self).__init__()
        self.input_features=input_features
        self.output_features=output_features
        self.fc1=nn.Linear(self.input_features,90)
        #self.bn1=nn.BatchNorm1d(90)
        self.fc2=nn.Linear(90,45)
        #self.bn2=nn.BatchNorm1d(45)
        self.fc3=nn.Linear(45,20)
        #self.bn3=nn.BatchNorm1d(20)
        self.fc4=nn.Linear(20,self.output_features)
        self.relu=nn.ReLU()

    def forward(self,X):
        output=self.relu((self.fc1(X)))
        output=self.relu((self.fc2(output)))
        output=self.relu((self.fc3(output)))
        output=self.fc4(output)
        return output


class EPDADataset(Dataset):
    def __init__(self,X_array,y_array,transform=None):
        super(EPDADataset,self).__init__()
        self.X_array=X_array
        self.y_array=y_array
        self.transform=transform

    def __len__(self):
        return self.X_array.shape[0]

    def __getitem__(self,idx):
        sample={
            'X':self.X_array[idx],
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






