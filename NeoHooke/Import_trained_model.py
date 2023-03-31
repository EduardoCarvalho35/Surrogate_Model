import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sympy as sym
import numpy as np
from matplotlib import pyplot as plt
import timeit
import pickle

n_inputs=9
n_outputs=9
n_samples=50000000

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
    
d = {}
with open(f"Info_tune_and_training{n_samples}.txt") as f:
    for line in f:
        print(line)
        (key, val) = line.split(maxsplit=1)
        d[key] = val.strip()


model = SurrogateModel(float(d["learning_rate:"]),list(map(int, list((d["Output_Layers_Dimensions:"][1:-1]).split(", ")))))
with open(f"SurrogateModel{n_samples}.pickle", "rb") as fp:
    model.load_state_dict(pickle.load(fp))
    
print(model)
# print(model(torch.tensor(np.array([1.2,0,0,0,1.2**(-0.5),0,0,0,1.2**(-0.5)]),dtype=torch.float32)).detach().numpy())
# # print(model.all_layers[0].weight)

def forward_pass_relu_fortran(trained_model,n_layers):
    features=sym.Matrix(sym.MatrixSymbol("x", trained_model[0].in_features,1))

    outputs={}
    outputs_sym={}
    for i in range(n_layers):
        if i==0:
            outputs[f"pass{i+1}"]=(sym.Matrix(trained_model[2*i].weight.detach().numpy())*features+sym.Matrix(trained_model[2*i].bias.detach().numpy()))
        else:
            outputs[f"pass{i+1}"]=sym.Matrix(trained_model[2*i].weight.detach().numpy())*outputs_sym[f"pass{i}"]+sym.Matrix(trained_model[2*i].bias.detach().numpy())
        if i!=(n_layers-1):
            outputs_sym[f"pass{i+1}"]=(sym.MatrixSymbol(f"pass{i+1}outputRelu", outputs[f"pass{i+1}"].shape[0],1))

    # outputs[f"pass{n_layers}"]=outputs[f"pass{n_layers}"].transpose()
    # return (outputs[f"pass{n_layers}"])
    return (outputs,outputs_sym)

file = open("ML_NO_BC.for", "w")
n_layers=1+int(d["n_layers:"])
file.write(f"      SUBROUTINE ML_NO_BC(DFGRD,pass{n_layers})\n      \n      IMPLICIT NONE\n      DOUBLE PRECISION x(9,1)\n      DOUBLE PRECISION DFGRD(3,3)\n      INTEGER i,j\n      ")
for i in range(n_layers):
    aux=forward_pass_relu_fortran(model.all_layers,n_layers)[0][f"pass{i+1}"].shape
    file.write(f"DOUBLE PRECISION pass{i+1}{aux}\n      ")
    print(i)
for i in range(len(forward_pass_relu_fortran(model.all_layers,n_layers)[1])):
    aux=forward_pass_relu_fortran(model.all_layers,n_layers)[0][f"pass{i+1}"].shape
    file.write(f"DOUBLE PRECISION pass{i+1}outputRelu{aux}\n      ")
    print(i)
file.write("\n      \n      ")
file.write("do i=1,3\n            do j=1,3\n                  x(j+3*(i-1),1)=DFGRD(i,j)\n            end do\n      end do\n      \n")
for i in range(len(forward_pass_relu_fortran(model.all_layers,n_layers)[0])):
    file.write((sym.fcode(forward_pass_relu_fortran(model.all_layers,n_layers)[0][f"pass{i+1}"],assign_to=f"pass{i+1}")))
    file.write("\n      \n")
    print(i)
    if (i+1)<n_layers:
        file.write(f"\n      do i=1,SIZE(pass{i+1}, DIM = 1)\n            if (pass{i+1}(i,1)<0) THEN\n                  pass{i+1}(i,1)=0\n            end if\n            pass{i+1}outputRelu(i,1)=pass{i+1}(i,1)\n      end do\n      \n")
        print(i)
file.write("      END SUBROUTINE ML_NO_BC \n")
file.close()
