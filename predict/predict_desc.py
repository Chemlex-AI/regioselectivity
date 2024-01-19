import os
import pickle

import pandas as pd
from rdkit import Chem
from .handler import ReactivityDescriptorHandler
from tqdm import tqdm

from .post_process import check_chemprop_out, min_max_normalize

def reaction_to_reactants(reactions):
    reactants = set()
    for r in reactions:
        rs = r.split('>')[0].split('.')
        reactants.update(set(rs))
    return list(reactants)


def predict_desc(t_data,v_data,args, in_train = True, normalize=True):

    def num_atoms_bonds(smiles):
        m = Chem.MolFromSmiles(smiles)

        m = Chem.AddHs(m)

        return len(m.GetAtoms()), len(m.GetBonds())

    
    reactivity_data = pd.concat([t_data, v_data])
    
    reactants = reaction_to_reactants(reactivity_data['rxn_smiles'].tolist())
 
    print('Predicting descriptors for reactants...')

    handler = ReactivityDescriptorHandler()

    descs = []
    for  smiles in tqdm(reactants):
        
        descs.append(handler.predict(smiles))

    df = pd.DataFrame(descs)
    
    invalid = check_chemprop_out(df)
    # FIXME remove invalid molecules from reaction dataset
    print(invalid)

    

    df.to_pickle('reactants_descriptors.pickle')

    if not normalize:
        return df

    if in_train:
        df, scalers = min_max_normalize(df)
        pickle.dump(scalers, open('models/scalers.pickle', 'wb'))
    else:
        scalers = pickle.load(open('models/scalers.pickle', 'rb'))
        df, _ = min_max_normalize(df, scalers=scalers)

    df.to_pickle('reactants_descriptors_norm.pickle')

    return df
