import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sympy as sym
import numpy as np
from matplotlib import pyplot as plt
import timeit

#### Load features and labels

train_features=np.load('train_features.npy',allow_pickle=True) 
train_labels=np.load('train_labels.npy',allow_pickle=True) 

test_features=np.load('test_features.npy',allow_pickle=True) 
test_labels=np.load('test_labels.npy',allow_pickle=True) 

train_size=train_features.shape[0]
test_size=test_features.shape[0]

####Flatten 3*3 2D array to get a 1D array of len 9

train_features=np.reshape(np.vstack(train_features),(train_size,9))
train_labels=np.reshape(np.vstack(train_labels),(train_size,9))

test_features=np.reshape(np.vstack(test_features),(test_size,9))
test_labels=np.reshape(np.vstack(test_labels),(test_size,9))



#### Hyperparameters

validation_split = 0.2
learning_rate = 0.0005
BATCH_SIZE = 50
EPOCHS = 20
n_layers=3
n_inputs=train_features.shape[1]
n_outputs=train_labels.shape[1]


#### Create torch tensors with featuers and labels
class MyDataset():
 
  def __init__(self,features,labels):
    
    self.features=torch.tensor(features,dtype=torch.float32)
    self.labels=torch.tensor(labels,dtype=torch.float32)

 
  def __len__(self):
    return len(self.labels)
   
  def __getitem__(self,idx):
    return self.features[idx],self.labels[idx]

train_data=MyDataset(train_features,train_labels)
test_data=MyDataset(test_features,test_labels)

#### SPLIT TRAIN DATA TO VALIDATION AND TRAIN DATA

train_set_size = int(len(train_data) * (1-validation_split))
valid_set_size = len(train_data) - train_set_size

seed = torch.Generator().manual_seed(42)
train_set, valid_set = torch.utils.data.random_split(train_data, [train_set_size, valid_set_size], generator=seed)

#### CREATE DATALOADERS

train_loader=DataLoader(train_set,num_workers=48,batch_size=BATCH_SIZE,shuffle=True)
valid_loader=DataLoader(valid_set,num_workers=48,batch_size=BATCH_SIZE,shuffle=False)
test_loader=DataLoader(test_data,num_workers=48,batch_size=BATCH_SIZE,shuffle=False)

#### Model definition

class SurrogateModel(pl.LightningModule):
    def __init__(self,learning_rate,n_inputs,n_outputs):
        super(SurrogateModel,self).__init__()        
        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(n_inputs, 256),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            # 3rd hidden layer
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            # 4th hidden layer
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(32, n_outputs),
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        logits = self.all_layers(x)
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        features, labels = train_batch
        logits = self.forward(features)
        loss = F.mse_loss(logits,labels)
#         self.log("train_loss", loss,prog_bar=True)
        self.logger.experiment.add_scalars('loss', {'train': loss},self.current_epoch)
        return loss
    
    def validation_step(self, valid_batch, batch_idx):
        features, labels = valid_batch
        logits = self.forward(features)
        loss = F.mse_loss(logits,labels)
        self.logger.experiment.add_scalars('loss', {'valid': loss},self.current_epoch) 
#         self.log("valid_loss", loss, prog_bar=True, on_step=False, on_epoch=True)        
        
    def test_step(self, test_batch, batch_idx):
        features, labels = test_batch
        logits = self.forward(features)
        loss = F.mse_loss(logits,labels)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)      
        
model = SurrogateModel(learning_rate,n_inputs,n_outputs)
print(model)
 
 
start = timeit.default_timer()
trainer = pl.Trainer(max_epochs=EPOCHS,accelerator="gpu", devices=2)
trainer.fit(model,train_loader,valid_loader)
trainer.test(model, dataloaders=test_loader)
end = timeit.default_timer()

print(model(torch.tensor(np.array([1.2,0,0,0,1.2**(-0.5),0,0,0,1.2**(-0.5)]),dtype=torch.float32)).detach().numpy())

#### Forward Pass function    
# def forward_pass_relu_fortran(trained_model,n_layers):
#     features=sym.Matrix(sym.MatrixSymbol("x", trained_model[0].in_features,1))

#     outputs={}
#     outputs_sym={}
#     for i in range(n_layers):
#         if i==0:
#             outputs[f"pass{i+1}"]=(sym.Matrix(trained_model[2*i].weight.detach().numpy())*features+sym.Matrix(trained_model[2*i].bias.detach().numpy()))
#         else:
#             outputs[f"pass{i+1}"]=sym.Matrix(trained_model[2*i].weight.detach().numpy())*outputs_sym[f"pass{i}"]+sym.Matrix(trained_model[2*i].bias.detach().numpy())
#         if i!=(n_layers-1):
#             outputs_sym[f"pass{i+1}"]=(sym.MatrixSymbol(f"pass{i+1}outputRelu", outputs[f"pass{i+1}"].shape[0],1))

#     # outputs[f"pass{n_layers}"]=outputs[f"pass{n_layers}"].transpose()
#     # return (outputs[f"pass{n_layers}"])
#     return (outputs,outputs_sym)



#### Write fortran subroutine to use in umat to get stress from deformation gradient 

# file = open("ML_BC.for", "w")
# n_layers=3
# file.write(f"      SUBROUTINE ML(DFGRD,pass{n_layers})\n      \n      IMPLICIT NONE\n      DOUBLE PRECISION x(9,1)\n      DOUBLE PRECISION DFGRD(3,3)\n      INTEGER i,j\n      ")
# for i in range(n_layers):
#     aux=forward_pass_relu_fortran(model.all_layers,n_layers)[0][f"pass{i+1}"].shape
#     file.write(f"DOUBLE PRECISION pass{i+1}{aux}\n      ")
#     print(i)
# for i in range(len(forward_pass_relu_fortran(model.all_layers,n_layers)[1])):
#     aux=forward_pass_relu_fortran(model.all_layers,n_layers)[0][f"pass{i+1}"].shape
#     file.write(f"DOUBLE PRECISION pass{i+1}outputRelu{aux}\n      ")
#     print(i)
# file.write("\n      \n      ")
# file.write("do i=1,3\n            do j=1,3\n                  x(j+3*(i-1),1)=DFGRD(i,j)\n            end do\n      end do\n      \n")
# for i in range(len(forward_pass_relu_fortran(model.all_layers,n_layers)[0])):
#     file.write((sym.fcode(forward_pass_relu_fortran(model.all_layers,n_layers)[0][f"pass{i+1}"],assign_to=f"pass{i+1}")))
#     file.write("\n      \n")
#     print(i)
#     if (i+1)<n_layers:
#         file.write(f"\n      do i=1,SIZE(pass{i+1}, DIM = 1)\n            if (pass{i+1}(i,1)<0) THEN\n                  pass{i+1}(i,1)=0\n            end if\n            pass{i+1}outputRelu(i,1)=pass{i+1}(i,1)\n      end do\n      \n")
#         print(i)
# file.write("      END SUBROUTINE ML")
# file.close()

#### Checking if forward pass function gives the same result obtained with pytorch model

# [outputs,outputs_sym]=forward_pass_relu_fortran(model.all_layers,n_layers)
# x=sym.MatrixSymbol("x", model.all_layers[0].in_features,1)
# outputs["pass1"]=sym.Matrix(outputs["pass1"].subs({x:sym.Matrix(test_features[0])}))
# for i in range(len(outputs_sym)):
#     outputs[f"pass{i+1}"]=sym.Matrix(outputs[f"pass{i+1}"])
#     for j in range(outputs[f"pass{i+1}"].shape[0]):
#         outputs[f"pass{i+1}"][j,0]=sym.Heaviside(outputs[f"pass{i+1}"][j,0])*outputs[f"pass{i+1}"][j,0]
#     aux=outputs_sym[f"pass{i+1}"]
#     outputs[f"pass{i+2}"]=outputs[f"pass{i+2}"].subs({aux:outputs[f"pass{i+1}"]}).as_mutable()

#     # aux=outputs_sym[f"pass{i+1}"]
# print(outputs["pass3"])
# print(model(torch.tensor(test_features,dtype=torch.float32)).detach().numpy()[0])

