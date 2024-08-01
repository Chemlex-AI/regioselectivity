import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gather_nd(t1, t2, max_nei):
    new_t = torch.zeros_like(t2, dtype = torch.float)
    new_t.resize_(t2.size()[0],t2.size()[1], t2.size()[2], t1.size()[-1])
    for i1 in range(max_nei):
        rows = t2[:,:,i1,0]
        columns = t2[:,:,i1,1]
        new_t[:,:,i1,:] = t1[rows[:,:],columns[:,:]]
    return new_t

def mask_mat(t, idx):
    range_tensor = torch.arange(idx).to(t.device)
    mask_tensor = range_tensor < t.unsqueeze(2)
    output_tensor = mask_tensor.int().unsqueeze(-1)
    return output_tensor

class MultiHeadedAttentionWithGate(nn.Module):
    
    def __init__(self, heads, d_model, max_nei, dropout=0.1):
        super(MultiHeadedAttentionWithGate, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)
        self.max_nei = max_nei
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model * 2, d_model)
        self.value = torch.nn.Linear(d_model * 2, d_model)
        self.nei_gate = torch.nn.Linear(d_model, d_model)
        self.atom_max = torch.nn.Linear(d_model * 2, d_model)
        self.gate = torch.nn.Linear(self.d_k * 4, 1)
        
    def forward(self, input_multihead, input_q):
        query = self.query(input_q)
        key = self.key(input_multihead)        
        value = self.value(input_multihead)   
        
        query = query.view(query.shape[0], self.heads, -1, self.d_k)   
        key = key.view(key.shape[0], self.heads, -1, self.max_nei, self.d_k)
        value = value.view(value.shape[0], self.heads, -1, self.max_nei, self.d_k)  
        score = torch.einsum('bhad,bhand->bhan', query, key)
        score = F.softmax(score,dim=-1).unsqueeze(-1)
        attention = score * value
        attention = attention.sum(dim = -2)
        
        ave = torch.mean(input_multihead.view(input_multihead.shape[0], self.heads, -1, self.max_nei, self.d_k*2), dim = -2)
        max_nr = self.atom_max(input_multihead)
        max_nr = max_nr.view(max_nr.shape[0], self.heads, -1, self.max_nei, self.d_k)
        element_max, max_index = torch.max(max_nr, dim= -2)
        cur_atom = input_q.view(input_q.shape[0], self.heads, -1, self.d_k) 
        gate_input = torch.cat([cur_atom, element_max, ave], dim = -1)
        gate_score = self.gate(gate_input)
        gate_score = F.softmax(gate_score, dim = 1)
        
        attention = gate_score * attention
        
        attention = attention.contiguous().view(attention.shape[0], -1, self.heads * self.d_k)
        return attention          

class MPN(nn.Module):
    
    def __init__(self, hidden_size, depth, max_nei=10):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_nei = max_nei
        self.attention = MultiHeadedAttentionWithGate(4, hidden_size, max_nei)
        self.atom_features = nn.Linear(34, hidden_size)
        self.nei_atom = nn.Linear(hidden_size, hidden_size)
        self.nei_bond = nn.Linear(46, hidden_size)
        self.self_atom = nn.Linear(hidden_size, hidden_size)
        self.label_U2 = nn.Linear(hidden_size + 46, hidden_size)
        self.label_U1 = nn.Linear(hidden_size * 2, hidden_size)
        self.activate = nn.ReLU()
        
    def forward(self, graph_inputs):
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, _, _ = graph_inputs
        
        atom_features = self.atom_features(input_atom)
        layers = []
        fbond_nei = gather_nd(input_bond, bond_graph.long(), self.max_nei)
        h_nei_bond = self.nei_bond(fbond_nei)
        
        for i in range(self.depth):
            
            fatom_nei = gather_nd(atom_features, atom_graph.long(), self.max_nei)           
            h_nei_atom = self.nei_atom(fatom_nei)
            h_nei = h_nei_atom * h_nei_bond    
            mask_nei = mask_mat(num_nbs.long(), self.max_nei)
            f_nei = (h_nei * mask_nei).sum(dim=-2)
            f_self = self.self_atom(atom_features)
            
            layers.append(f_nei * f_self * node_mask.unsqueeze(-1))
            l_nei = torch.cat([fatom_nei, fbond_nei], dim=3)
            pre_label = self.label_U2(l_nei)
            pre_label = self.activate(pre_label)
          
            nei_label = pre_label * mask_nei
            nei_att  = torch.cat([nei_label, h_nei_bond], dim=3)
            f_attention = self.attention(nei_att, atom_features)

            new_label = torch.cat([atom_features, f_attention], dim=2)
        
            atom_features = self.label_U1(new_label)
            atom_features = self.activate(atom_features)
        kernels = torch.mean(torch.cat([l.unsqueeze(3) for l in layers], dim=3),3)

        return kernels
    
