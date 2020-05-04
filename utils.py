import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class GetDataset(Dataset):
    def __init__(self, X, Y):

        self.X = X   # numpy.ndarray (num_data, num_nodes(N), T_in, num_features(F))
        self.Y = Y   # numpy.ndarray (num_data, num_nodes(N), T_out, num_features(F))
        
    def __getitem__(self, index):
        
        # torch.Tensor
        tensor_X = self.X[index]
        tensor_Y = self.Y[index]
        
        return tensor_X, tensor_Y

    def __len__(self):
        return len(self.X)

def make_dataset(rawdata, T_in, T_out):

    """
    :input: rawdata (num_nodes(N), T, F1)
    :return X: (num_data, num_nodes(N), T_in, F2)
    :return Y: (num_data, num_nodes(N), T_out)
    """
    X,Y = [],[] 
    T_all = rawdata.shape[1]
    pdata = rawdata.copy()
    # be used to discretize y for PA approximation (y has been normalized to [0,1])
    pdata = np.concatenate([((pdata[:,:,:1]-1e-8)*50).astype(int),pdata],axis=-1) 
    for i in range(T_all-(T_in+T_out)+1):
        X.append(pdata[:, i:i+T_in, :])
        Y.append(rawdata[:, i+T_in:i+(T_in+T_out), :1])
    X = torch.from_numpy(np.asarray(X)).float()
    Y = torch.from_numpy(np.asarray(Y)).float().squeeze(-1)
    print('X shape',X.shape)
    print('Y shape',Y.shape)
    return GetDataset(X, Y)

def adj_process(adj,train_num,topk,disteps):
    """
    return: sparse CxtConv and sparse PropConv adj
    """
    # sparse context graph adj (2,E)
    edge_1 = []
    edge_2 = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if(i==j or (adj[i,j]<=disteps)):
                edge_1.append(i)
                edge_2.append(j)
    edge_adj = np.asarray([edge_1,edge_2],dtype=int)
    
    # sparse propagating adj (2,E)
    edge_1 = []
    edge_2 = [] 
    for i in range(adj.shape[0]):
        cnt = 0
        adj_row = adj[i,:train_num]
        adj_row = sorted(enumerate(adj_row), key=lambda x:x[1])  # [(idx,dis),...]
        for j,dis in adj_row:
            if(i!=j):  
                edge_1.append(i)
                edge_2.append(j)
                cnt += 1
            if(cnt >= topk and dis>disteps):
                break
    adj_label = np.asarray([edge_1,edge_2],dtype=int)
    return edge_adj, adj_label

def load_data(T_in, T_out, Batch_Size, train_num, topk, disteps):
    # adjacency matrix
    adj = np.load('../data/adj.npy') 
    print('adj shape:',adj.shape) # (N, N)
    # parking availability dataset (including PA, time and contextual data)
    padata = np.load('../data/padata.npy') #(N, T_all, F)
    N, T_all, _ = padata.shape
    print('X shape:',padata.shape)  # (N, T_all, F)
    # total parking spots
    total_park = np.load('../data/total_park.npy') # (N, )
    dataset_train = make_dataset(padata[:,:int(T_all*0.6)],T_in,T_out)
    print('len of dataset_train:',len(dataset_train))
    dataset_val = make_dataset(padata[:,int(T_all*0.6):int(T_all*0.8)],T_in,T_out)
    print('len of dataset_val:',len(dataset_val))
    dataset_test = make_dataset(padata[:,int(T_all*0.8):],T_in,T_out)
    print('len of dataset_test:',len(dataset_test))
    loader_train = DataLoader(dataset=dataset_train, batch_size=Batch_Size, shuffle=True, pin_memory=True,num_workers=1)
    loader_val = DataLoader(dataset=dataset_val, batch_size=Batch_Size, shuffle=False, pin_memory=True,num_workers=1)
    loader_test = DataLoader(dataset=dataset_test, batch_size=Batch_Size, shuffle=False, pin_memory=True,num_workers=1)
    idx_train = range(0,train_num) # labeled parking lots
    idx_val = range(train_num, N) # unlabeled parking lots
    # adj process
    adjs = adj_process(adj,train_num,topk,disteps) # return CxtConv and PropConv adj
    print("load_data finished.")
    return adjs,loader_train,loader_val,loader_test,idx_train,idx_val,total_park.reshape(1,N,1)
    


