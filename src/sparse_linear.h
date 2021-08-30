#pragma once
#include <torch/extension.h>
#ifndef TORCH_V1_5
  #include <ATen/record_function.h>
#else
  #include <torch/csrc/autograd/record_function.h>
#endif
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include "sparse_tensor.h"
#include "sparse_gemm.h"

static torch::Tensor _bsc_forward(torch::Tensor input,
                              int shape[2],
                              torch::Tensor weight,
                              torch::Tensor rowidx,
                              torch::Tensor colptr,
                              int ncolptr,
                              torch::Tensor bias,
                              bool with_relu,
                              int block_size[2]
                              ){
  ;
  int M = input.size (0);
  assert (input.size (1) == weight.size (0));
  int K = input.size (1);
  int N = weight.size (1);
  auto output = at::empty({M * N}, input.options());
  if (input.scalar_type() == at::kFloat){
      sparse_gemm_bsc_f32(M, N, K, input.data_ptr<float>(), weight.data_ptr<float>(), rowidx.data_ptr<int64_t>(),
                          colptr.data_ptr<int64_t>(), ncolptr, block_size, output.data_ptr<float>(), with_relu);
  }
  else{
      printf("bf16 TBD");
  }
  return output;
};

class SparseLinearOp : public torch::autograd::Function<SparseLinearOp> {
public:
static torch::Tensor forward (torch::Tensor input,
                              SparseTensor weight,
                              int bk, int bn, string name,
                              torch::Tensor bias,
                              string attr)
{
  int M = input.size (0);
  assert (input.size (1) == weight.shape[0]);
  int K = input.size (1);
  int N = weight.shape[1];
  int blocksize[2] = { bk, bn };
  auto shape = weight.shape;
  auto sparse_tensor = weight.sparse_tensor;
  assert(int(weight.format) == 1);
  auto rowidx = weight.index_tensors[0];
  auto colptr = weight.index_tensors[1];
  return _bsc_forward(input, shape, sparse_tensor, rowidx, colptr, colptr.size(0), bias, attr=="relu", blocksize);
}
};
