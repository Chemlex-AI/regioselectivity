import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ..graph_utils.mol_graph import smiles2graph_pr, pack1D, pack2D, pack2D_withidx, get_mask
from ..graph_utils.ioutils_direct import binary_features_batch

class GraphDataset(Dataset):
    def __init__(self, smiles, products, rxn_id, predict=False):
        self.smiles = smiles
        self.products = products
        self.rxn_id = rxn_id
        self.atom_classes = {}
        self.predict = predict

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        r = self.smiles[index]
        ps = self.products[index]
        rxn_id = self.rxn_id[index]
        if not self.predict:
            x, y = self.__data_generation(r, ps, rxn_id)
            return x, y
        else:
            x = self.__data_generation(r, ps, rxn_id)
            return x

    def __data_generation(self, smiles_temp, products_temp, rxn_id_temp):
        size = len(products_temp.split('.'))
        rxn_id_extend = [rxn_id_temp]*size
        prs_extend = [smiles2graph_pr(p, smiles_temp, core_buffer=0) for p in products_temp.split('.')]
        labels_extend = [1] + [0] * (size - 1)

        rs_extends, smiles_extend = zip(*prs_extend)
        fatom_list, fatom_qm_list, fbond_list, gatom_list, gbond_list, nb_list, core_mask = zip(*rs_extends)
        
        res_graph_inputs = [
            fatom_list,
            fbond_list,
            gatom_list,  
            gbond_list,
            nb_list,
            fatom_list,
            smiles_extend,
            core_mask,
            fatom_qm_list,
        ]
      
        if self.predict:
            return res_graph_inputs
        else:
            return res_graph_inputs, labels_extend

class GraphDataLoader(DataLoader):
    def __init__(self, smiles, products, rxn_id, batch_size, predict=False, collate_fn = None, num_workers = -1):
        dataset = GraphDataset(smiles, products, rxn_id, predict=predict)
        super(GraphDataLoader, self).__init__(dataset, batch_size=batch_size, collate_fn = collate_fn, num_workers = num_workers)

