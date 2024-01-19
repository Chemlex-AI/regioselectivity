import torch
from MPNN.graph_utils.mol_graph import pack2D, pack1D, pack2D_withidx, get_mask
from MPNN.graph_utils.ioutils_direct import binary_features_batch
import numpy as np
import torch.nn as nn
from numpy import linalg
from steric_hinderance import mol_conf, get_randm_points, vdw_r
from rdkit import Chem

stm = nn.Softmax(dim=0)

def train_collate_batch(batch_data):
    data, targets = zip(*batch_data)
    data_ls = []
    dict_f = {}

    for ili in data:
        for ix, ilil in enumerate(ili):
            for ilili in ilil:
                if ix not in dict_f.keys():
                    
                    dict_f[ix] = [ilili]
                else:
                    dict_f[ix].append(ilili)

    data_ls = [
        torch.tensor(pack2D(dict_f[0]), dtype= torch.float),
        torch.tensor(pack2D(dict_f[1]), dtype= torch.float),
        torch.tensor(pack2D_withidx(dict_f[2]), dtype= torch.float),
        torch.tensor(pack2D_withidx(dict_f[3]), dtype= torch.float),
        torch.tensor(pack1D(dict_f[4]), dtype= torch.float),
        torch.tensor(get_mask(dict_f[5]), dtype= torch.long),
        torch.tensor(binary_features_batch(dict_f[6]), dtype= torch.float),
        torch.tensor(pack1D(dict_f[7]), dtype= torch.float),
        torch.tensor(pack2D(dict_f[8]), dtype= torch.float),
    ]

    target_ls= []
    for i in targets:
        target_ls.extend(i)
    target_ls = torch.tensor(target_ls).long().unsqueeze(-1)
    
    return data_ls, target_ls

def test_collate_batch(batch_data):
    data = batch_data
    data_ls = []
    dict_f = {}

    for ili in data:
        for ix, ilil in enumerate(ili):
            for ilili in ilil:
                if ix not in dict_f.keys():
                    
                    dict_f[ix] = [ilili]
                else:
                    dict_f[ix].append(ilili)
    
    data_ls = [
        torch.tensor(pack2D(dict_f[0]), dtype= torch.float),
        torch.tensor(pack2D(dict_f[1]), dtype= torch.float),
        torch.tensor(pack2D_withidx(dict_f[2]), dtype= torch.float),
        torch.tensor(pack2D_withidx(dict_f[3]), dtype= torch.float),
        torch.tensor(pack1D(dict_f[4]), dtype= torch.float),
        torch.tensor(get_mask(dict_f[5]), dtype= torch.long),
        torch.tensor(binary_features_batch(dict_f[6]), dtype= torch.float),
        torch.tensor(pack1D(dict_f[7]), dtype= torch.float),
        torch.tensor(pack2D(dict_f[8]), dtype= torch.float),
    ]
    return data_ls

def custom_loss(yh_pred, yh_gt):
    flat_gt = yh_gt.view(-1)
    flat_pred = yh_pred.view(-1)
    ones = torch.where(flat_gt==1)[0].to('cpu').tolist()
    loss = 0
    
    for ii,__ in enumerate(ones):
        
        try:
            loss += torch.sum(torch.log(torch.clamp(stm(torch.sub(flat_pred[ones[ii]:ones[ii+1]], torch.max(flat_pred[ones[ii]:ones[ii+1]]))), min = 1e-9, max = 1-1e-9)) * flat_gt[ones[ii]:ones[ii+1]] * torch.tensor(-1), dtype=float)
        except:
            loss += torch.sum(torch.log(torch.clamp(stm(torch.sub(flat_pred[ones[ii]:], torch.max(flat_pred[ones[ii]:]))), min = 1e-9, max = 1-1e-9)) * flat_gt[ones[ii]:] * torch.tensor(-1), dtype=float)
    return loss/flat_gt.size()[0]
    
    

def regio_accuracy(y_pred,y_gt):
    corr = 0
    count = 0
    for aa, bb in zip(y_pred, y_gt):
       
        flat_gt = np.array(bb.view(-1).to('cpu').tolist())
        flat_pred = aa.view(-1).to('cpu').tolist()

        indss = np.where(flat_gt == 1)
        leng = indss[0].size
        count += leng
        
        for lab in range(leng):
            lab_pred = indss[0][lab]
            
            if lab == leng - 1:
                ls_sub = list(flat_pred[lab_pred:])
            else:
                lab_pred1 = indss[0][lab+1]
            
                ls_sub = list(flat_pred[lab_pred:lab_pred1])
            if ls_sub.index(max(ls_sub)) == 0:
                corr += 1
    return corr/count 

def numpy_sh_perct(smiles, center, n):
    s = mol_conf(smiles)
    
    try:
        dff = s[0][0]
    except:
        return None
    
    res = []
    atom_coord = np.array(dff[['x','y','z']],dtype = np.float64)
    radius = np.array([vdw_r(i) for i in dff['symbol']])
    for i in range(atom_coord.shape[0]):
        mc_samples = np.array(get_randm_points(atom_coord[i],center,n))
        ex_atom_3 = np.broadcast_to(atom_coord,(n,atom_coord.shape[0],atom_coord.shape[1]))
        ex_mc_sample = np.broadcast_to(mc_samples,(atom_coord.shape[0], n,atom_coord.shape[1]))
        ex_mc_sample = np.swapaxes(ex_mc_sample,0,1)
        sub_matrix = ex_atom_3 - ex_mc_sample
        dis_matrix = linalg.norm(sub_matrix,axis=2)
        rdw_matrix = np.broadcast_to(radius, (n,radius.shape[0]))
        ss = rdw_matrix - dis_matrix
        ss = np.where(ss>0,1,0)
        ss[:,i]=0
        ss = np.sum(ss,axis=1)
        ss = np.where(ss>0,1,0)
    
        res.append(np.sum(ss)/n)
    return res