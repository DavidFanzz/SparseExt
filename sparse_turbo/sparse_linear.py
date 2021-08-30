import random
import math
import torch
import torch.nn as nn
from _C import mlp_sparse_forward
from _C import SparseTensor
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SparseFC(Function):
    use_sparse_kernel = True

    @staticmethod
    def forward(ctx, input, weight, bias, use_sparse_kernels=False, k=1, b=16, name="sparselinear", attr="none"):

        ctx.save_for_backward(input)
        ctx.save_for_backward(weight)
        ctx.save_for_backward(bias)
        if use_sparse_kernels:
            output = mlp_sparse_forward(input, weight.t(), k, b, name, bias, attr)
            SparseFC.use_sparse_kernels = True
        else:
            output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class SparseLinear(torch.nn.Module):
    def __init__(self, C, K, bias=None, bk=1, bn=16, name='sparselinear'):
        super(SparseLinear, self).__init__()
        self.C = C
        self.K = K 
        self.bk = bk
        self.bn = bn
        self.weight = Parameter(torch.Tensor(K, C))
        if bias:
            self.bias = Parameter(torch.Tensor(K))
        else:
            self.register_parameter('bias', None)
        self.name = name
        self._sparse_weight = None
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.C * self.K)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        sparsity = int(torch.sum(self.weight == 0.)) / self.weight.numel()
        use_sparse_kernels = True if sparsity > 0.6 else False
        if self._sparse_weight is None:
            self._sparse_weight = SparseTensor(self.name+'_weight',
                                               self.weight,
                                               0,
                                               [self.bk, self.bn])
        output = SparseFC.apply(input, self._sparse_weight, self.bias, use_sparse_kernels, self.bk, self.bn, self.name)
        return output

class SparseLinearRelu(torch.nn.Module):
    def __init__(self, C, K, bias=None, bk=1, bn=16, name='sparselinear'):
        super(SparseLinearRelu, self).__init__()
        self.C = C
        self.K = K
        self.bk = bk
        self.bn = bn
        self.weight = torch.nn.Parameter(
            torch.empty(K, C))
        if bias:
            self.bias = Parameter(torch.Tensor(K))
        else:
            self.register_parameter('bias', None)
        self.name = name
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.C * self.K)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        sparsity = int(torch.sum(self.weight == 0.)) / self.weight.numel()
        use_sparse_kernels = True if sparsity > 0.6 else False
        output = SparseFC.apply(input, self.weight, self.bias, use_sparse_kernels, self.bk, self.bn, self.name, "relu")
        return output


if __name__ == "__main__":
    model = SparseLinear(1024, 512, bias=True)
    weight = torch.randn(1024, 512)
    for i in range(1024):
        for j in range(512//16):
            if random.random() > 0.1:
                weight[i, 16*j:(16*j+16)] = 0
    bias = torch.zeros(512)
    model.weight.data.copy_(weight)
    model.bias.data.copy_(bias)
    input = torch.randn([32768, 1024])
    y = model(input)
    weight_ = torch.transpose(weight, 0, 1)
    linear = nn.Linear(1024, 512)
    linear.weight.data.copy_(weight_)
    linear.bias.data.copy_(bias)
    for _ in range(10):
        y1 = model(input)
        y2 = linear(input)

    import time
    begin1 = time.time()
    for _ in range(1000):
        y1 = model(input)
    print("sparse comp time: {}".format(time.time() - begin1))

    begin2 = time.time()
    for _ in range(1000):
        y2 = linear(input)
    print("dense comp time: {}".format(time.time() - begin2))
