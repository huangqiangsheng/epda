from __future__ import print_function,division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Evaluator(object):
    def __init__(self,loss=nn.MSELoss(),batch_size=64):
        self.loss=loss
        self.batch_size=batch_size

    def evaluate(self,model,data,device=torch.device('cpu')):
        model.eval()
        loss=self.loss
        dataloader= DataLoader(dataset=data,batch_size=self.batch_size,shuffle=True,num_workers=0)
        _loss_val=0.0
        rmse=0
        with torch.no_grad():
            model=model.to(device)
            for batch in dataloader:
                input_vars=batch['X'].to(device)
                target_vars=batch['y'].to(device)
                outputs=model(input_vars)
                _loss_val+=loss(outputs,target_vars).item()
        return _loss_val/(len(dataloader)),rmse

