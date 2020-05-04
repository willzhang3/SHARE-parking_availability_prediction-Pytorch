import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from hierarchical_graph_conv import GAT, SCConv

class FeatureEmb(nn.Module):
    def __init__(self):
        super(FeatureEmb, self).__init__()
        # time embedding 
        # month,day,hour,minute,dayofweek
        self.time_emb = nn.ModuleList([nn.Embedding(feature_size, 4) for feature_size in [12,31,24,4,7]])
        for ele in self.time_emb:
            nn.init.xavier_uniform_(ele.weight.data, gain=math.sqrt(2.0))
        
    def forward(self, X, pa_onehot):
        B, N, T_in, F = X.size() # (batch_size, N, T_in, F)
        X_time =torch.cat([emb(X[:,:,:,i+4].long()) for i,emb in enumerate(self.time_emb)],dim=-1) # time F = 4*5 = 20
        X_cxt = X[...,2:4] # contextual features
        X_pa = X[...,:1].long() # PA, 0,1,...,49
        pa_scatter = pa_onehot.clone()
        X_pa = pa_scatter.scatter_(-1,X_pa,1.0) # discretize to one-hot , F = 50 
        return X_cxt, X_pa, X_time
    
class SHARE(nn.Module):
    def __init__(self,args,t_in,t_out,latend_num,train_num,dropout=0.5,alpha=0.2,hid_dim=32,\
                 gat_hop=2,device=torch.device('cuda')):
        super(SHARE, self).__init__()
        self.device = device
        # number of context features (here set 2 for test)
        self.nfeat = 2 
        self.hid_dim = hid_dim
        self.train_num = train_num
        # Feature embedding
        self.feature_embedding = FeatureEmb() 
        # FC layers
        self.output_fc = nn.Linear(hid_dim*2, t_out, bias=True)
        self.w_pred = nn.Linear(hid_dim*2, 50, bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Spatial blocks
        # CxtConv
        self.CxtConv = GAT(in_feat=self.nfeat, nhid=hid_dim, dropout=dropout, alpha=alpha, hopnum=gat_hop, pa_prop=False)
        # PropConv
        self.PropConv = GAT(in_feat=self.nfeat, nhid=hid_dim, dropout=dropout, alpha=alpha, hopnum=1, pa_prop=True)
        # SCConv
        self.SCConv = SCConv(in_features=hid_dim+50, out_features=hid_dim, dropout=dropout,\
                                   alpha=alpha, latend_num=latend_num, gcn_hop = 1)
        
        # GRU Cell
        self.GRU = nn.GRUCell(2*hid_dim+50+20, hid_dim*2, bias=True)
        nn.init.xavier_uniform_(self.GRU.weight_ih,gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU.weight_hh,gain=math.sqrt(2.0))
        
        # Parameter initialization
        for ele in self.modules():
            if isinstance(ele, nn.Linear):
                nn.init.xavier_uniform_(ele.weight,gain=math.sqrt(2.0))
    
    def forward(self, adjs, X, h_t, pa_onehot):
        """
        :param adjs: CxtConv, PropConv and SCconv adj.
        :param X: Input data of shape (batch_size, num_nodes(N), T_in, num_features(F)).
        :param h_t: To init GRU hidden state with shape (N, 2*hid_dim).
        :param pa_onehot: be used to discretize y for PA approximation
        :return: predicted PA and CE_loss
        """
        adj,adj_label,adj_dense = adjs
        B,N,T,F_feat = X.size()
        X_cxt,X_pa,X_time = self.feature_embedding(X, pa_onehot)
        # GRU and Spatial blocks
        CE_loss = 0.0 
        for i in range(T):
            y_t = F.softmax(self.w_pred(h_t),dim=-1) # (B, N, p=50)
            if(i==T-1):
                CE_loss += F.binary_cross_entropy(y_t[:,:self.train_num,:].reshape(B*self.train_num,-1),\
                        X_pa[:,:self.train_num,i,:].reshape(B*self.train_num,-1))
            # PropConv
            y_att = self.PropConv(X_cxt[:,:,i,:],X_pa[:,:,i,:], adj_label) # (B, N, p=50)
            if(i==T-1):
                y_att[:,:self.train_num,:] = torch.where(y_att[:,:self.train_num,:]<1.,y_att[:,:self.train_num,:],\
                                                         (1.-1e-8)*torch.ones(1,1).cuda())
                CE_loss += F.binary_cross_entropy(y_att[:,:self.train_num,:].reshape(B*self.train_num,-1),\
                        X_pa[:,:self.train_num,i,:].reshape(B*self.train_num,-1))
            # PA approximation
            en_yt = torch.exp(torch.sum(y_t*torch.log\
                                        (torch.where(y_t>1e-8,y_t,1e-8*torch.ones(1,1).cuda())),dim=-1,keepdim=True)) 
            en_yatt = torch.exp(torch.sum(y_att*torch.log\
                                        (torch.where(y_att>1e-8,y_att,1e-8*torch.ones(1,1).cuda())),dim=-1,keepdim=True))
            en_yatt = torch.where(torch.sum(y_att,dim=-1,keepdim=True)>1e-8,en_yatt,torch.zeros(1,1).cuda())
            pseudo_y = (en_yt*y_t + en_yatt*y_att)/(en_yt+en_yatt)
            if(self.training == False):
                pseudo_y[:,:self.train_num,:] = X_pa[:,:self.train_num,i,:]
            # CxtConv
            h_cxt = self.CxtConv(X_cxt[:,:,i,:],None,adj) # (B, N, tmp_hid)
            # SCConv
            h_sc = self.SCConv(torch.cat([h_cxt,pseudo_y],dim=-1),adj_dense)
            X_feat = torch.cat([h_cxt,pseudo_y,h_sc,X_time[...,i,:]],dim=-1)
            h_t = self.GRU(X_feat.view(-1,2*self.hid_dim+50+20), h_t.view(-1,self.hid_dim*2)) # (B*N, 2*tmp_hid)
            h_t = h_t.view(B,N,-1)
            
        out = torch.sigmoid(self.output_fc(h_t)) # (B, N, T_out)
        return out, CE_loss


