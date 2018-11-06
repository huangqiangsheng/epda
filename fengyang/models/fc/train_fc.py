import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import argparse
import logging
import torch
from torchvision import transforms
from util.trainer import SupervisedTrainer
from util.checkpoint import Checkpoint
import torch.optim as optim
import torch.nn as nn
from models.fc.fc import EPDADataset,ToTensor,FCNet
from sklearn.model_selection import train_test_split
from util.optim import Optimizer
raw_input = input  # Python 3
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./checkpoints',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--batch_size',default=256,dest='batch_size',type=int)
parser.add_argument('--device',default='cpu',dest='device')
parser.add_argument('--lr',default=1e-4,dest='lr',type=float)
parser.add_argument('--weight_decay',default=0.,dest='weight_decay',type=float)
opt = parser.parse_args()


LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

device=torch.device('cpu')
if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model.to(device)
else:
    input_features=51*6
    data = pd.read_csv('../../data/data.csv')
    X = data.iloc[:, :input_features]
    for col in X.columns:
        if col.startswith('te1_') or col.startswith('te2_') or col.startswith('tm1_') or col.startswith('tm2_'):
            X[col] = np.log(X[col])
    X_describe=X.describe()
    X_describe.to_csv('X_describe.csv',index=True)
    y_indices = [i for i in range(input_features + 3, input_features + 3 + 4)] + [i for i in range(input_features+12,input_features+16)]
    y = data.iloc[:, y_indices]
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)
    y=scaler.fit_transform(y)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train = EPDADataset(np.array(X_train),np.array(y_train),transform=transforms.Compose([ToTensor()]))
    dev = EPDADataset(np.array(X_test),np.array(y_test),transform=transforms.Compose([ToTensor()]))
    loss = nn.MSELoss()

    model = None
    optimizer = None
    output_features=y_train.shape[1]
    if not opt.resume:
        model =FCNet(input_features,output_features).to(device)
        for param in model.parameters():
            param.data.uniform_(-0.08,0.08)
    print('model:',model)
    t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size,
                          checkpoint_every=1000,
                          print_every=100, expt_dir=opt.expt_dir,device=device)
    optimizer=Optimizer(optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay), max_grad_norm=5)
    #scheduler = StepLR(optimizer.optimizer, 30,gamma=0.5)
    #optimizer.set_scheduler(schedule)
    seq2seq = t.train(model, train,
                      num_epochs=100000, dev_data=dev,lr=opt.lr,
                      optimizer=optimizer,
                      resume=opt.resume)



