import pandas as pd
from MPNN.MPNN.data_loading import GraphDataLoader
from MPNN.graph_utils.mol_graph import initialize_qm_descriptors
from predict.predict_desc import predict_desc
import argparse
import torch
from MPNN.MPNN.model import MPNN
from tqdm import tqdm
from utils import test_collate_batch, numpy_sh_perct
from rdkit import Chem
from rdkit.Chem import AllChem

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', default = 20, type = int)
parser.add_argument('-hs', '--hidden_size', default = 100, type = int)
parser.add_argument('-d', '--depth', default = 4, type = int)
parser.add_argument('-m', '--model_path', default='model/model_structure.pt', type = str)
parser.add_argument('-test', '--test_path', required=True, type = str)
args = parser.parse_args()
   
device = torch.device('cuda')

test = pd.read_csv(args.test_path)
test_rxn_id = test['reaction_id'].values
test_smiles = test.rxn_smiles.str.split('>', expand=True)[0].values
test_products = test.products_run.values
test_types = test.reaction_type.values

df = predict_desc(test, pd.DataFrame(), args, in_train=False)
initialize_qm_descriptors(df=df)
test_data = GraphDataLoader(test_smiles, test_products, test_rxn_id, args.batch_size, predict=True, collate_fn = test_collate_batch, num_workers=10)

model = MPNN(args.hidden_size,args.depth)
model.load_state_dict(torch.load(args.model_path))
model = model.to(device)

model.eval()
predss = []
# model inference reacting score
for t_data in tqdm(test_data):
    test_p = []
    for tensors in t_data:
        if not isinstance(tensors, list):
            test_p.append(tensors.squeeze().to(device))
        else:
            test_p.append(tensors)
    preds = model(test_p)
    predss.append(preds.to('cpu').tolist())


Buchwald_template = '[#7;!H0:1].[Cl,Br,I,$(OS(=O)(=O))][#6:2]>>[#7:1]-[#6:2]'
Heck_template = "[C:1]=[C:2][#6:3].[Cl,Br,I,$(OS(=O)(=O)C(F)(F)F),$(OS(=O)(=O)c1ccc([CH3])cc1),$([N+]#N)][#6:4]>>[#6:4][C:1]=[C:2][#6:3]"
Hiyama_template = "[#6:1][Si]([CH3,F,O])([CH3,F,O])[CH3,F,O].[Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)][#6:2]>>[#6:1][#6:2]" 
Kumada_template = "[#6:1][Mg][Cl,Br,I].[F,Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)][#6:2]>>[#6:1][#6:2]"
Negishi_template = "[#6:1][Zn][Cl,Br,I].[Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)][#6:2]>>[#6:1][#6:2]"
Sonogashira_template = "[#6:1][C:2]#[CH1:3].[F,Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)]-[#6:4]>>[#6:1][C:2]#[C:3][#6:4]"
Stille_template = "[#6:1][Sn]([#6])[#6].[F,Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)][#6:2]>>[#6:1][#6:2]"
Suzuki_template = "B(O)(O)([#6:1]).[Cl,Br,I,$(OS(=O)(=O)C(F)(F)F),$(O[S](=[O])(=[O])c1ccc([CH3])cc1),$(O[S](=[O])(=[O])[CH3])][#6:2]>>[#6:1]-[#6:2]"

res_dict = {'reaction_id': test_rxn_id, 'predicted_main_product': []}

num = 0

for indice, (test_product, test_smile,test_type) in tqdm(enumerate(zip(test_products,test_smiles,test_types))):
    tps = test_product.split('.')
    length = len(tps)
    sterics = []
    pros = None
    for tp in tps:
        rxn = AllChem.ReactionFromSmarts(eval(f'{test_type}_template'))
        reactant1, reactant2 = test_smile.split('.')
        mol1,mol2 = Chem.MolFromSmiles(reactant1), Chem.MolFromSmiles(reactant2)
        pros = rxn.RunReactants([mol1, mol2])
        # find the reacting halogen group
        if pros:
            steric = numpy_sh_perct(reactant2, center=5, n=6000)
            if not steric:
                sterics.append(0) 
            for pro in pros:
                if pro[0].HasSubstructMatch(Chem.MolFromSmiles(tp)):
                    for atoms in pro[0].GetAtoms():
                        if 'old_mapno' in atoms.GetPropsAsDict().keys() and (test_type !='Heck' and test_type != 'Sonogashira'):
                            if atoms.GetPropsAsDict()['old_mapno'] == 2:
                                atom_idx = atoms.GetPropsAsDict()['react_atom_idx']
                                sterics.append(steric[atom_idx])
                                break
                        elif 'old_mapno' in atoms.GetPropsAsDict().keys() and (test_type =='Heck' or test_type == 'Sonogashira'):
                            if atoms.GetPropsAsDict()['old_mapno'] == 4:
                                atom_idx = atoms.GetPropsAsDict()['react_atom_idx']
                                sterics.append(steric[atom_idx])
                                break
        else:
            pros = rxn.RunReactants([mol2, mol1])
            steric = numpy_sh_perct(reactant1, center=5, n=6000)
            if not steric:
                sterics.append(0) 
            for pro in pros:
                if pro[0].HasSubstructMatch(Chem.MolFromSmiles(tp)):
                    for atoms in pro[0].GetAtoms():
                        if 'old_mapno' in atoms.GetPropsAsDict().keys() and (test_type !='Heck' and test_type != 'Sonogashira'):
                            if atoms.GetPropsAsDict()['old_mapno'] == 2:
                                atom_idx = atoms.GetPropsAsDict()['react_atom_idx']
                                sterics.append(steric[atom_idx])
                                break
                        elif 'old_mapno' in atoms.GetPropsAsDict().keys() and (test_type =='Heck' or test_type == 'Sonogashira'):
                            if atoms.GetPropsAsDict()['old_mapno'] == 4:
                                atom_idx = atoms.GetPropsAsDict()['react_atom_idx']
                                sterics.append(steric[atom_idx])
                                break
    it, res = indice//args.batch_size, indice%args.batch_size
    scores = predss[it][num:num+length]
    if res == args.batch_size-1:
        num = 0
    else:
        num += length  
    s_s = zip(sterics, scores, tps)
    s_s = sorted(s_s, key = lambda x: x[1], reverse = True)
    for ss in s_s:
        if ss[0] <= 0.6:
            res_dict['predicted_main_product'].append(ss[2])
            break
    
    # add the product with the highest reacting score for reaction with the steric of products are all greater than 0.6
    if len(res_dict['predicted_main_product']) == indice:
        res_dict['predicted_main_product'].append(s_s[0][2])
        
pd.DataFrame.from_dict(res_dict).to_csv('example_data/inference_result.csv', index=False)
