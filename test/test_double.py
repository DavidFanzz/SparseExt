import torch
import torch.nn as nn

import random
from sparse_mlp import SparseLinear

o_feature = 512
i_feature = 1024
model1 = SparseLinear(i_feature, o_feature, bias=True, name="1")
weight = torch.randn(o_feature, i_feature)
for i in range(i_feature):
    for j in range(o_feature//16):
        if random.random() > 0.001:
            weight[16*j:(16*j+16), i] = 0
# weight_ = weight.transpose(0, 1)
bias = torch.zeros(o_feature)
model1.weight.data.copy_(weight)
model1.bias.data.copy_(bias)
input = torch.randn([32768, i_feature])
linear1 = nn.Linear(i_feature, o_feature)
linear1.weight.data.copy_(weight)
linear1.bias.data.copy_(bias)
for _ in range(10):
    y1 = model1(input)
    y2 = linear1(input)
    print((y1-y2).max())

o_feature = 256
i_feature = 1024
model2 = SparseLinear(i_feature, o_feature, bias=True, name="2")
weight = torch.randn(o_feature, i_feature)
for i in range(i_feature):
    for j in range(o_feature//16):
        if random.random() > 0.001:
            weight[16*j:(16*j+16), i] = 0
# weight_ = weight.transpose(0, 1)
bias = torch.zeros(o_feature)
model2.weight.data.copy_(weight)
model2.bias.data.copy_(bias)
input = torch.randn([32768, i_feature])
linear2 = nn.Linear(i_feature, o_feature)
linear2.weight.data.copy_(weight)
linear2.bias.data.copy_(bias)
for _ in range(10):
    y1 = model2(input)
    y2 = linear2(input)

for _ in range(100):
    y1_1 = model1(input)
    y1_2 = model2(input)
    y2_1 = linear1(input)
    y2_2 = linear2(input)
    print((y1_1 - y2_1).max())
    print((y1_2 - y2_2).max())
    
