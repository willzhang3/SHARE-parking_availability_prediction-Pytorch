import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import dgl
import torch.nn as nn
import logging
import random
from share import SHARE
from utils import *

which_gpu = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
parser = argparse.ArgumentParser(description='SHARE')
parser.add_argument('--enable_cuda', action='store_true', default=True, help='GPU training.')
parser.add_argument('--loss', type=str,default='mse', help='Loss function.')
parser.add_argument('--seed', type=int, default=33, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=4, help='Number of batch to train and test.')
parser.add_argument('--t_in', type=int, default=12, help='Input time step.')
parser.add_argument('--t_out', type=int, default=3, help='Output time step.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hid_dim', type=int, default=32, help='Dim of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hc_ratio', type=float, default=0.1, help='ratio in Hierarchical graph.')
parser.add_argument('--topk', type=int, default=10, help='k nearest neighbors in Propagating graph.')
parser.add_argument('--disteps', type=int, default=1000, help='Farthest neighbors distance in Context graph.')
parser.add_argument('--gat_hop', type=int, default=2, help='Number of hop in CxtConv.')
parser.add_argument('--train_num', type=int, default=-1, help='Number of parking lots for train.')
parser.add_argument('--train_ratio', type=float, default=0.3, help='Parking lots ratio for train.')
parser.add_argument('--patience', type=int, default=30, help='Patience')
parser.add_argument('--beta', type=float, default=0.5, help='Beta of CE_loss .')

args = parser.parse_args()
args.train_num = int(1965*args.train_ratio+0.5)
logging.basicConfig(level = logging.INFO,filename='./log',format = '%(asctime)s - %(process)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.enable_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print(args)
logger.info(args)

def train_epoch(loader_train,h_t,prr_onehot):
    """
    Trains one epoch with the given data.
    :param loader_train: Training data (X,Y).
    :return: Average loss of last bacth.
    """
    for i,(X_batch,Y_batch) in enumerate(loader_train):
        net.train()
        X_batch = X_batch.to(device=args.device)
        Y_batch = Y_batch.to(device=args.device)
        now_batch = X_batch.shape[0]
        optimizer.zero_grad()
        if(now_batch == args.batch_size):
            y_pred,prr_loss = net(adjs, X_batch, h_t, prr_onehot) # (B, N , T_out)
        else:
            adj,adj_label,adj_dense = adjs
            adjs_copy = (spadj_expand(g_adj,now_batch),spadj_expand(g_adj_label,now_batch),adj_dense[:now_batch])
            y_pred,prr_loss = net(adjs_copy,X_batch,h_t[:now_batch],prr_onehot[:now_batch])
        loss = loss_criterion(y_pred[:,idx_train,:], Y_batch[:,idx_train,:]) + prr_loss*args.beta 
        loss.backward()
        optimizer.step()
        if (i*args.batch_size % 80 == 0):
            print(i*args.batch_size)
            print("train loss:{:.4f}".format(loss.detach().cpu().numpy()))
    return loss.detach().cpu().numpy()

def test_epoch(loader_val,h_t,prr_onehot):
    """
    Test one epoch with the given data.
    :param loader_val: Valuation or Test data (X,Y).
    :return: Loss and MAE.
    """
    val_loss = []
    val_mae = []
    for i,(X_batch,Y_batch) in enumerate(loader_val):
        if (i*args.batch_size % 80 == 0):
            print(i*args.batch_size)
        net.eval()
        X_batch = X_batch.to(device=args.device)
        Y_batch = Y_batch.to(device=args.device) # (B,N,T_out)
        now_batch = X_batch.shape[0]
        if(now_batch == args.batch_size):
            y_pred,prr_loss = net(adjs, X_batch, h_t, prr_onehot) # (B,N,T_out)
        else:
            adj,adj_label,adj_dense = adjs
            adjs_copy = (spadj_expand(g_adj,now_batch),spadj_expand(g_adj_label,now_batch),adj_dense[:now_batch])
            y_pred,prr_loss = net(adjs_copy, X_batch, h_t[:now_batch], prr_onehot[:now_batch])
        loss_val = loss_criterion(y_pred[:,idx_val,:], Y_batch[:,idx_val,:]) + prr_loss*args.beta
        val_loss.append(np.asscalar(loss_val.detach().cpu().numpy()))
        mae = np.absolute(y_pred[:,idx_val,:].detach().cpu().numpy()*total_park[:,idx_val]\
                          -Y_batch[:,idx_val,:].detach().cpu().numpy()*total_park[:,idx_val]) # (B,N,T_out)
        val_mae.append(mae)
    return np.asarray(val_loss),np.concatenate(val_mae,axis=0)
        
def spadj_expand(adj, batch_size):
    adj = dgl.batch([adj]*batch_size)
    return adj

def print_log(mae,mse,loss,stage):
    mae_o = [np.mean(mae[:,:,i]) for i in range(args.t_out)]
    mse_o = [np.mean(mse[:,:,i]) for i in range(args.t_out)]
    rmse_o = [np.sqrt(ele) for ele in mse_o]
    stage_str = "{} - mean metrics: mae,mse,rmse,loss".format(stage)
    mean_str = "mean metric values: {},{},{},{}".format(np.mean(mae_o),np.mean(mse_o),np.mean(rmse_o),np.mean(loss))
    mae_str = "MAE: {}".format(','.join(str(ele) for ele in mae_o))
    mse_str = "MSE: {}".format(','.join(str(ele) for ele in mse_o))
    rmse_str = "RMSE: {}".format(','.join(str(ele) for ele in rmse_o))
    print(stage_str)
    print(mean_str)
    print(mae_str)
    print(mse_str)
    print(rmse_str)
    logger.info(stage_str)
    logger.info(mean_str)
    logger.info(mae_str)
    logger.info(mse_str)
    logger.info(rmse_str)
        
if __name__ == '__main__':
    
    adjs,loader_train,loader_val,loader_test,idx_train,idx_val,total_park = \
            load_data(args.t_in,args.t_out,args.batch_size,args.train_num,args.topk,args.disteps)
    N = total_park.shape[1] # total number of parking lots
    adj,adj_label = adjs
    latend_num = int(N*args.hc_ratio+0.5) # latent node number
    print('latend num:',latend_num)
    adj_edgenum = adj.shape[1]
    adj_label_edgenum = adj_label.shape[1]
    adj = torch.from_numpy(adj).long()
    adj_label = torch.from_numpy(adj_label).long()
    # Merge 2 graph as scconv's adj
    adj_dense = torch.sparse_coo_tensor(adj,torch.ones((adj.shape[1])),torch.Size([N,N])).to_dense()
    adj_dense_label = torch.sparse_coo_tensor(adj_label,torch.ones((adj_label.shape[1])),torch.Size([N,N])).to_dense()
    adj_dense = adj_dense + adj_dense_label
    adj_dense = torch.where(adj_dense<1e-8,adj_dense,torch.ones(1,1))
    adj_merge = adj_dense.to(device=args.device).repeat(args.batch_size,1,1)
    g_adj = dgl.DGLGraph()
    g_adj.add_nodes(N)
    g_adj.add_edges(adj[0],adj[1])
    # expand for batch training
    adj = spadj_expand(g_adj,args.batch_size)
    g_adj_label = dgl.DGLGraph()
    g_adj_label.add_nodes(N)
    g_adj_label.add_edges(adj_label[0],adj_label[1])
    adj_label = spadj_expand(g_adj_label,args.batch_size)
    adjs = (adj,adj_label,adj_merge)
    # to init GRU hidden state
    h_t = torch.zeros(args.batch_size,N,args.hid_dim*2).to(device=args.device)
    # to discretize y for PA approximation
    prr_onehot = torch.zeros(args.batch_size,N,args.t_in,50).to(device=args.device)
    
    # model
    net = SHARE(args = args,
            t_in=args.t_in,
            t_out=args.t_out,
            latend_num = latend_num,
            train_num = args.train_num,
            dropout=args.dropout, 
            alpha=args.alpha, 
            hid_dim=args.hid_dim,
            gat_hop = args.gat_hop,
            device=args.device).to(device=args.device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    if(args.loss == 'mse'):
        loss_criterion = nn.MSELoss()
    elif(args.loss == 'mae'):
        loss_criterion = nn.L1Loss()
        
    best_epoch = 0
    min_mse = 1e15
    st_epoch = best_epoch
    for epoch in range(st_epoch,args.epochs):
        st_time = time.time()
        # training
        print('training......')
        loss_train = train_epoch(loader_train,h_t,prr_onehot)
        # validating
        with torch.no_grad():
            print('validating......')
            val_loss,val_mae = test_epoch(loader_val,h_t,prr_onehot)
            val_mse = val_mae**2
        # testing
        with torch.no_grad():
            print('testing......')
            test_loss,test_mae = test_epoch(loader_test,h_t,prr_onehot)
            test_mse = test_mae**2
            
        val_meanmse = np.mean(val_mse)
        if(val_meanmse < min_mse):
            min_mse = val_meanmse
            best_epoch = epoch + 1
            best_mae = test_mae.copy()
            best_mse = test_mse.copy()
            best_loss = test_loss.copy()
        # log
        try:
            print("Epoch: {}".format(epoch+1))
            logger.info("Epoch: {}".format(epoch+1))
            print("Train loss: {}".format(loss_train))
            logger.info("Train loss: {}".format(loss_train))
            print_log(val_mae,val_mse,val_loss,'Validation')
            print_log(test_mae,test_mse,test_loss,'Test')
            print_log(best_mae,best_mse,best_loss,'Best Epoch-{}'.format(best_epoch))
            print('time: {:.4f}s'.format(time.time() - st_time))
            logger.info('time: {:.4f}s\n'.format(time.time() - st_time))
        except:
            print("log error...")
            
        # early stop
        if(epoch+1 - best_epoch >= args.patience):
            sys.exit(0)
