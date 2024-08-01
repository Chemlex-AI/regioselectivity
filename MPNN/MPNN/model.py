import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MPN
from rxnfp.core import FingerprintGenerator
from transformers import BertModel
from rxnfp.transformer_fingerprints import get_default_model_and_tokenizer
import math 

model_rxn, tokenizer = get_default_model_and_tokenizer()

class rxn_bert(FingerprintGenerator):
    
    def __init__(self, model: BertModel):
        super(rxn_bert).__init__()
        self.model_bt = model
        self.model_bt.to('cuda:0')
        
    def convert(self, input, trainning=True):
        
        self.model_bt.eval()
        with torch.no_grad():
            output = self.model_bt(**input)
        test = output['last_hidden_state'][:,0,:]
        return test
    
    def convert_batch():
        None

bert = rxn_bert(model_rxn)


class MultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(256, d_model)
        self.value = torch.nn.Linear(256, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, input_multihead, input_q):
        query = self.query(input_q)
        key = self.key(input_multihead)        
        value = self.value(input_multihead)   
        
        query = query.view(query.shape[0], self.heads, 1, self.d_k)   
        key = key.view(key.shape[0], self.heads, 1, self.d_k)
        value = value.view(value.shape[0], self.heads, 1, self.d_k)  
        scores = torch.matmul(query, key.permute(0,1,3,2)) / math.sqrt(query.size(-1))
        
        weights = F.softmax(scores, dim=1)          
        weights = self.dropout(weights)

        context = torch.matmul(weights, value)

        context = context.contiguous().view(context.shape[0], self.heads * self.d_k)

        return self.output_linear(context)



class MPNN(nn.Module):
    def __init__(self, hidden_size, depth, max_nb=10):
        super(MPNN, self).__init__()
        self.hidden_size = hidden_size
        self.reactants_MPN = MPN(hidden_size, depth, max_nb)
        self.reaction_score0 = nn.Linear(hidden_size + 40, hidden_size, bias=False)
        self.reaction_score = nn.Linear(hidden_size * 2, 1)
        self.rxnfp_attention = MultiHeadedAttention(4, hidden_size)

    def forward(self, inputs):
        res_rss = inputs[-1]

        bert_inputs = tokenizer.batch_encode_plus(res_rss, 
                                            max_length=model_rxn.config.max_position_embeddings, 
                                            padding=True, truncation=True, return_tensors='pt').to('cuda:0')
        if self.training:
            res_rxnfp = bert.convert(bert_inputs, trainning = True)
        else:
            res_rxnfp = bert.convert(bert_inputs, trainning = False)
            
        res_inputs = inputs[:8]
        res_atom_mask = res_inputs[-3]
        res_core_mask = res_inputs[-1]
        fatom_qm = inputs[-2]
        res_atom_hidden = self.reactants_MPN(res_inputs)
        res_atom_hidden = F.relu(res_atom_hidden)
        res_atom_hidden = torch.cat([res_atom_hidden, fatom_qm], dim=-1)
        res_atom_mask = res_atom_mask.unsqueeze(-1)
        res_core_mask = res_core_mask.unsqueeze(-1)
        res_atom_hidden = self.reaction_score0(res_atom_hidden)
        res_mol_hidden = (res_atom_hidden * res_atom_mask * res_core_mask).sum(dim=-2)
        attention = self.rxnfp_attention(res_rxnfp, res_mol_hidden)
        reaction_score = self.reaction_score(torch.cat((res_mol_hidden, attention),1))
        
        
        return reaction_score