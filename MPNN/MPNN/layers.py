import torch
import torch.nn as nn
import torch.nn.functional as F

def gather_nd(t1, t2):
    new_t = torch.zeros_like(t2, dtype = torch.float)
    new_t.resize_(t2.size()[0],t2.size()[1], t2.size()[2], t1.size()[-1])
    for i1 in range(10):
        rows = t2[:,:,i1,0]
        columns = t2[:,:,i1,1]
        new_t[:,:,i1,:] = t1[rows[:,:],columns[:,:]]
    return new_t

def mask_mat(t, idx):
    range_tensor = torch.arange(idx).to(t.device)
    mask_tensor = range_tensor < t.unsqueeze(2)
    output_tensor = mask_tensor.int().unsqueeze(-1)
    return output_tensor
            

class MPN(nn.Module):
    
    def __init__(self, hidden_size, depth, max_nb=10):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_nb = max_nb

        self.atom_features = nn.Linear(34, hidden_size)
        self.nei_atom = nn.Linear(hidden_size, hidden_size)
        self.nei_bond = nn.Linear(40, hidden_size)
        self.self_atom = nn.Linear(hidden_size, hidden_size)
        self.label_U2 = nn.Linear(hidden_size + 40, hidden_size)
        self.label_U1 = nn.Linear(hidden_size * 2, hidden_size)
        self.activate = nn.ReLU()
        
    def forward(self, graph_inputs):
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, _, _ = graph_inputs

        atom_features = self.atom_features(input_atom)
        layers = []
        fbond_nei = gather_nd(input_bond, bond_graph.long())
        h_nei_bond = self.nei_bond(fbond_nei)
        
        for i in range(self.depth):
            
            fatom_nei = gather_nd(atom_features, atom_graph.long())           
            h_nei_atom = self.nei_atom(fatom_nei)
            h_nei = h_nei_atom * h_nei_bond    
            mask_nei = mask_mat(num_nbs.long(), self.max_nb)
            f_nei = (h_nei * mask_nei).sum(dim=-2)
            f_self = self.self_atom(atom_features)
         
            layers.append(f_nei * f_self * node_mask.unsqueeze(-1))
            l_nei = torch.cat([fatom_nei, fbond_nei], dim=3)
            pre_label = self.label_U2(l_nei)
            pre_label = self.activate(pre_label)
          
            nei_label = (pre_label * mask_nei).sum(dim=-2)
          
            new_label = torch.cat([atom_features, nei_label], dim=2)
        
            atom_features = self.label_U1(new_label)
            atom_features = self.activate(atom_features)

        kernels = layers[-1]
        return kernels
