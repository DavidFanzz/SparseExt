import torch
import torch.nn as nn
import time
import random
from sparse_mlp import SparseLinear

class SparseNet(nn.Module):
    def __init__(self, dims):
        super(SparseNet, self).__init__()
        self.dims = dims
        self.create_layers()

    def create_layers(self):
        layers = []
        layerNum = 0
        for dim in self.dims:
            if dim == self.dims[0]:
                layers.append(nn.Linear(dim[0], dim[1], bias=True))
            else:
                layers.append(SparseLinear(dim[0], dim[1], bias=True, name="{}".format(layerNum)))
            layerNum += 1
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, dims):
        super(DenseNet, self).__init__()
        self.dims = dims
        self.create_layers()

    def create_layers(self):
        layers = []
        for dim in self.dims:
            layers.append(nn.Linear(dim[0], dim[1], bias=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__=="__main__":
    dims = [[13, 512], [512, 1024], [1024, 256]]
    sparseModel = SparseNet(dims)
    denseModel = DenseNet(dims)
    sparsity = 0.99
    bs = 32768
    weights = {}
    weights_ = {}
    for layer in range(len(dims)):
        dim = dims[layer]
        if layer == 0:
            weight = torch.randn(dim[1], dim[0])
        else:
            weight = torch.randn(dim[1], dim[0])
            for i in range(dim[0]):
                for j in range(dim[1]//16):
                    if random.random() < sparsity:
                        weight[16*j:(16*j+16), i] = 0
        bias = torch.randn(dim[1])
        weights['layers.{}.weight'.format(layer)] = weight.clone().detach().requires_grad_()
        weights['layers.{}.bias'.format(layer)] = bias.clone().detach().requires_grad_()
        #if layer == 0:
        weights_['layers.{}.weight'.format(layer)] = weight.clone().detach().requires_grad_()
        #else:
        #weights_['layers.{}.weight'.format(layer)] = torch.transpose(weight.clone().detach().requires_grad_(), 0, 1)
        weights_['layers.{}.bias'.format(layer)] = bias.clone().detach().requires_grad_()
    denseModel.load_state_dict(weights_)
    sparseModel.load_state_dict(weights)
    input = torch.randn(bs, dims[0][0])
    for _ in range(20):
        y = sparseModel(input)
        z = denseModel(input)
    timeBegin = time.time()
    for _ in range(1000):
        y = sparseModel(input)
    print('Time sparse: %s s' % (time.time() - timeBegin))

    timeBegin = time.time()
    for _ in range(1000):
        z = denseModel(input)
    print('Time ref: %s s' % (time.time() - timeBegin)) 

