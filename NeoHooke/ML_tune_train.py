import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import argparse

import timeit
import pickle

# from pytorch_lightning.loggers import NeptuneLogger
# import neptune

# neptune_logger = NeptuneLogger(
#     api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyN2EzMzJmZC0wMzk2LTRmZjYtYjRkMi1lYTg0MmIxY2QyNTMifQ==",
#     project="eduardocarvalho/SurrogateModel-Geral")


# PARAMETERS
n_samples=50000000
n_inputs=9
n_outputs=9
test_split=0.01  #fraction of the dataset used for testing
val_split =0.01 #fraction of the dataset data used for valdiation
tuning_valid_percent=0.1  #fraction of the validation data used for tunning
tuning_train_percent=0.1  #fraction of the training data used for tunning
n_trials_tunning=300
timeout_tunning=3600
max_epochs=15


class MyDataset():
  def __init__(self,features,labels):
    self.features=torch.tensor(features,dtype=torch.float32)
    self.labels=torch.tensor(labels,dtype=torch.float32)
    
  def __len__(self):
    return len(self.labels)
   
  def __getitem__(self,idx):
    return self.features[idx],self.labels[idx]

def data_pre_process(n_samples,n_inputs,n_outputs,test_split,val_split):
    df=pd.read_pickle(f"GenerateDataframe_F_Geral_{n_samples}.pkl")

    train_df, test_df = train_test_split(df, test_size = test_split)

    train_size=train_df.shape[0]
    test_size=test_df.shape[0]
 
    train_features=np.reshape(np.vstack(train_df["F"].to_numpy()),(train_size,n_inputs))
    train_labels=np.reshape(np.vstack(train_df["SIGMA"].to_numpy()),(train_size,n_outputs))

    test_features=np.reshape(np.vstack(test_df["F"].to_numpy()),(test_size,n_inputs))
    test_labels=np.reshape(np.vstack(test_df["SIGMA"].to_numpy()),(test_size,n_outputs))

    train_data=MyDataset(train_features,train_labels)
    test_set=MyDataset(test_features,test_labels)
    
    val_split=val_split/(1-test_split)
    train_set_size = int(len(train_data) * (1-val_split))
    valid_set_size = len(train_data) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = torch.utils.data.random_split(train_data, [train_set_size, valid_set_size], generator=seed)
    return train_set,valid_set,test_set

train_set,valid_set,test_set=data_pre_process(n_samples,n_inputs,n_outputs,test_split,val_split)

##### Dataloaders
class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(train_set,batch_size=self.batch_size,shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(valid_set,batch_size=self.batch_size,shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(test_set,batch_size=self.batch_size,shuffle=False, pin_memory=True)
    
#### Model
class SurrogateModel(pl.LightningModule):
    def __init__(self,learning_rate, output_dims):
        super(SurrogateModel,self).__init__()
        layers = []
        
        input_dim = n_inputs
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, n_outputs))

        self.all_layers=nn.Sequential(*layers) 
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.all_layers(x)

    def training_step(self, train_batch, batch_idx):
        features, labels = train_batch
        logits = self.forward(features)
        loss = F.mse_loss(logits,labels)
        self.logger.experiment.add_scalars('loss', {'train': loss},self.current_epoch)
        self.log("train_loss", loss,on_epoch=True, on_step=False)
        return {'loss': loss}
    

    def validation_step(self, valid_batch, batch_idx):
        features, labels = valid_batch
        logits = self.forward(features)
        loss = F.mse_loss(logits,labels)
        self.log("val_loss", loss,on_epoch=True, on_step=False)
        self.logger.experiment.add_scalars('loss', {'valid': loss},self.current_epoch) 
           
        
    def test_step(self, test_batch, batch_idx):
        features, labels = test_batch
        logits = self.forward(features)
        loss = F.mse_loss(logits,labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True)   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer 


#### Optuna Objective function   
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    learning_rate = trial.suggest_float("learning_rate",1e-4, 1e-1,log=True)
    batch_size = trial.suggest_categorical("batch_size", [16,32,64,128,256])
    output_dims = [trial.suggest_int("n_units_l{}".format(i), 4, 264, log=True) for i in range(n_layers)]

    model = SurrogateModel(learning_rate,output_dims)
    datamodule = DataModule(batch_size=batch_size)
    logger = TensorBoardLogger("tune_logs", name=f"SurrogateModel-Geral{n_samples}",log_graph=True)


    trainer = pl.Trainer(logger=logger,limit_val_batches=tuning_valid_percent,
        limit_train_batches=tuning_train_percent,
        accelerator="gpu",devices=1,strategy="dp",
        max_epochs=max_epochs)

    hyperparameters = dict(n_layers=n_layers,learning_rate=learning_rate,output_dims=output_dims,batch_size=batch_size)

    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()  
        
if __name__ == "__main__":
    file = open(f"Info_tune_and_training{n_samples}.txt", "w")
    start_tune = timeit.default_timer()
    
    parser = argparse.ArgumentParser(description="PyTorch Lightning distributed data-parallel training example.")
    parser.add_argument("--pruning","-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",)
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner())

    storage = "sqlite:///SurrogateModelGeral.db"
    study = optuna.create_study(
        study_name="3",
        storage=storage,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,)
    study.optimize(objective, n_trials=n_trials_tunning, timeout=timeout_tunning)

    print("Number of finished trials: {}".format(len(study.trials)))
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        file.write("{}: {}\n".format(key, value))
        
    end_tune = timeit.default_timer()
    print('Time tune (min): ',(end_tune-start_tune)/60)
        
    aux=[]
    for i in range(trial.params["n_layers"]):
        aux.append(trial.params[f"n_units_l{i}"])
    file.write(f"Output_Layers_Dimensions: {aux}\n")
    
    
    model = SurrogateModel(trial.params["learning_rate"],aux)
    datamodule = DataModule(batch_size=trial.params["batch_size"])
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5, mode="min")
    print(model)
    start = timeit.default_timer()
    
    logger = TensorBoardLogger("train_logs", name=f"SurrogateModel-Geral{n_samples}",log_graph=True)
    trainer = pl.Trainer(logger=logger,
        accelerator="gpu",devices=1,strategy="dp",
        callbacks=[early_stop_callback])

    trainer.fit(model,datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    end = timeit.default_timer()
    
    print('Time train (min): ',(end-start)/60)
    
    file.write(f'Time_tune(min): {(end_tune-start_tune)/60}\n')
    file.write(f'Time_train(min): {(end-start)/60}\n')

    print('Stress: ',model(torch.tensor(np.array([1.2,0,0,0,1.2**(-0.5),0,0,0,1.2**(-0.5)]),dtype=torch.float32)).detach().numpy())
    file.close()
    with open(f"SurrogateModel{n_samples}.pickle", "wb") as fp:
        pickle.dump(model.state_dict(), fp)
        
    # optuna-dashboard sqlite:///SurrogateModelGeral.db

        


 