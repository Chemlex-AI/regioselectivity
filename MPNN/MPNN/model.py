import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MPN

class MPNN(nn.Module):
    def __init__(self, hidden_size, depth, max_nb=10):
        super(MPNN, self).__init__()
        self.hidden_size = hidden_size
        self.reactants_MPN = MPN(hidden_size, depth, max_nb)

        self.reaction_score0 = nn.Linear(hidden_size + 40, hidden_size, bias=False)
        self.reaction_score = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        res_inputs = inputs[:8]
        res_atom_mask = res_inputs[-3]
        res_core_mask = res_inputs[-1]
        fatom_qm = inputs[-1]
        res_atom_hidden = self.reactants_MPN(res_inputs)
        res_atom_hidden = F.relu(res_atom_hidden)
        res_atom_hidden = torch.cat([res_atom_hidden, fatom_qm], dim=-1)
        res_atom_mask = res_atom_mask.unsqueeze(-1)
        res_core_mask = res_core_mask.unsqueeze(-1)
        res_atom_hidden = self.reaction_score0(res_atom_hidden)
        res_mol_hidden = (res_atom_hidden * res_atom_mask * res_core_mask).sum(dim=-2)
        reaction_score = self.reaction_score(res_mol_hidden)
        
        
        return reaction_score