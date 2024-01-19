## Regioselectivity for Metal-Catalyzed Cross-Coupling Reactions using Regio-MPNN 

## Conda environment
Conda environment can be prepared with environment.yml using:
```
conda env create -f environment.yml
```

## Training Phase
Start to train the model:
```
python train.py -ne number_of_training_epochs -m model_path
```

## Inference Phase
Inference:
```
python inference.py -m model_path
```