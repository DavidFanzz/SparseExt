import torch
import torch.nn as nn

import random
from sparse_turbo import SparseLinear
from torch.nn.parameter import Parameter
o_feature = 512
i_feature = 1024
model = SparseLinear(i_feature, o_feature, bias=True)
weight = torch.randn(o_feature, i_feature)
P_weight = Parameter(torch.Tensor(o_feature, i_feature))
for i in range(i_feature):
    for j in range(o_feature//16):
        if random.random() > 0.01:
            weight[16*j:(16*j+16), i] = 0
# weight_ = weight.transpose(0, 1)
bias = torch.zeros(o_feature)
model.weight = 
model.bias.data.copy_(bias)
input = torch.randn([32768, i_feature])
y = model(input)
# weight_ = torch.transpose(weight, 0, 1)
linear = nn.Linear(i_feature, o_feature)
linear.weight.data.copy_(weight)
linear.bias.data.copy_(bias)
for _ in range(10):
    y1 = model(input)
    y2 = linear(input)
print((y1-y2).max())


