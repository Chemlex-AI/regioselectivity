import pandas as pd
from MPNN.MPNN.data_loading import GraphDataLoader
from MPNN.graph_utils.mol_graph import initialize_qm_descriptors
from predict.predict_desc import predict_desc
import argparse
import torch
from torch.optim import Adam, lr_scheduler
from MPNN.MPNN.model import MPNN
from tqdm import tqdm
from utils import train_collate_batch, custom_loss, regio_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', default = 20, type = int)
parser.add_argument('-hs', '--hidden_size', default = 100, type = int)
parser.add_argument('-d', '--depth', default = 4, type = int)
parser.add_argument('-ne', '--n_epochs', default = 5, type = int)
parser.add_argument('-m','--model_path', default = 'models/model.pt', type = str) 
parser.add_argument('-lr', '--learning_rate', default = 1e-3, type= float)
args = parser.parse_args()
   
device = torch.device('cuda')

train = pd.read_csv('./demo_data/train_add_split.csv', index_col=0)
train_rxn_id = train['reaction_id'].values
train_smiles = train.rxn_smiles.str.split('>', expand=True)[0].values
train_products = train.products_run.values

valid = pd.read_csv('./demo_data/valid_add_split.csv', index_col=0)
valid_rxn_id = valid['reaction_id'].values
valid_smiles = valid.rxn_smiles.str.split('>', expand=True)[0].values
valid_products = valid.products_run.values

df = predict_desc(train, valid, args, in_train=True)
initialize_qm_descriptors(df=df)
train_data = GraphDataLoader(train_smiles, train_products, train_rxn_id, args.batch_size, predict=False, collate_fn = train_collate_batch, num_workers=10)
valid_data = GraphDataLoader(valid_smiles, valid_products, valid_rxn_id, args.batch_size, predict=False, collate_fn = train_collate_batch, num_workers=10)

model = MPNN(args.hidden_size,args.depth)
model = model.to(device)

optimizer = Adam(model.parameters(), lr = args.learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size = int(1000/args.batch_size), gamma=0.95)
n_epochs = args.n_epochs

for epoch in tqdm(range(n_epochs)):
    model.train()
    batch_loss = 0
    gts = []
    predss = []
    nums = 1 
    for t_data, t_lab in tqdm(train_data):
        
        t_p =[]
        for tensors in t_data:
            t_p.append(tensors.squeeze().to(device))
        st_lab = t_lab.to(device)

        model.zero_grad()
        preds = model(t_p)
        
        loss = custom_loss(preds.float(), st_lab.long())
        batch_loss += loss.item()
        
        gts.append(st_lab.long())
        predss.append(preds.float())
        
        loss.backward()        
        optimizer.step()
        scheduler.step()

        accuracy = regio_accuracy(predss, gts)
    
        print('Training Loss:', round(batch_loss/nums, 4))
        print('Current epoch Cumulative Training Set Accuracy:', round(accuracy, 4))
        nums += 1
    
    model.eval()
    batch_loss = 0
    gts = []
    predss = []
    
    for v_data, v_lab in tqdm(valid_data):
        
        v_p =[]
        for tensors in v_data:
            v_p.append(tensors.squeeze().to(device))
        sv_lab = v_lab.to(device)

        preds = model(v_p)
        
        loss = custom_loss(preds.float(), sv_lab.long())
        batch_loss += loss.item()
                
        gts.append(sv_lab.long())
        predss.append(preds.float())
        
    accuracy = regio_accuracy(predss, gts)
    
    print('Validation Loss:', round(batch_loss/len(valid_data), 4))
    print('Validation Set Accuracy:', round(accuracy, 4))
    
    if epoch == 0:
        torch.save(model.state_dict(), args.model_path)
        current_valid = accuracy
        
    elif accuracy > current_valid:
        torch.save(model.state_dict(), args.model_path)
        current_valid = accuracy
        