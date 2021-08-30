#pragma once

#include <assert.h>
#include <string.h>
#include <torch/extension.h>


enum SparseCompression{
    BSR,
    BSC
};

bool compress_coo(torch::Tensor& cs, int64_t* indices, int64_t dim, int64_t nnz){
  if (cs.size(0) != dim + 1){
    return false;
  }

  // TODO: eliminate this conditional when zero-size dims supported correctly
  if (nnz > 0) {
    auto cs_accessor = cs.accessor<int64_t, 1>();
    // Convert the sparse matrix to CSR format
    at::parallel_for(0, nnz, 10000, [&](int64_t start, int64_t end) {
      int64_t h, hp0, hp1;
      for (auto i = start; i < end; i++) {
        hp0 = indices[i];
        hp1 = (i+1 == nnz) ?  dim : indices[i+1];
        if (hp0 != hp1) {
          for (h = hp0; h < hp1; h++) {
            cs_accessor[h+1] = i+1;
          }
        }
      }
    });
  }
  else{
    return false;
  }
  return true;
}

void create_bsr_tensors(torch::Tensor& dense_tensor,
                        torch::Tensor& sparse_tensor,
                        std::vector<torch::Tensor>& index_tensors,
                        const int block_size[2],
                        const int shape[2]){
    auto reshaped_tensor = dense_tensor.reshape({dense_tensor.size(0) / block_size[0],
                                                 dense_tensor.size(1) / block_size[1],
                                                 block_size[0] * block_size[1]});
    auto values_indices = at::max(reshaped_tensor.abs(), -1);
    torch::Tensor compressed_values = std::get<0>(values_indices);
    torch::Tensor indices = at::nonzero(compressed_values);
    auto nnz = indices.size(0);
    auto dim = indices.size(1);
    torch::Tensor row_indices = indices.select(1, 0);
    torch::Tensor col_indices = indices.select(1, 1);
    torch::Tensor compressed_row_idx = at::zeros({dim + 1}, kLong);
    bool find_csr = compress_coo(compressed_row_idx, row_indices.contiguous().data_ptr<int64_t>(), dim, nnz);
    assert(find_csr);
    torch::Tensor mask = at::unsqueeze(at::not_equal(compressed_values, 0), -1).repeat({1, 1, block_size[0] * block_size[1]}) ;
    sparse_tensor = at::masked_select(reshaped_tensor, mask);
    index_tensors.push_back(col_indices);
    index_tensors.push_back(compressed_row_idx);
}

void create_bsc_tensors(torch::Tensor& dense_tensor,
                        torch::Tensor& sparse_tensor,
                        std::vector<torch::Tensor>& index_tensors,
                        const int block_size[2],
                        const int shape[2]){
    auto reshaped_tensor = dense_tensor.reshape({dense_tensor.size(0) / block_size[0],
                                                 dense_tensor.size(1) / block_size[1],
                                                 block_size[0] * block_size[1]});
    auto values_indices = at::max(reshaped_tensor.abs(), -1);
    torch::Tensor compressed_values = std::get<0>(values_indices);
    torch::Tensor indices = at::nonzero(compressed_values);
    auto nnz = indices.size(0);
    auto dim = indices.size(1);
    torch::Tensor row_indices = indices.select(1, 0);
    torch::Tensor col_indices = indices.select(1, 1);
    torch::Tensor compressed_col_idx = at::zeros({dim + 1}, kLong);
    bool find_csr = compress_coo(compressed_col_idx, col_indices.contiguous().data_ptr<int64_t>(), dim, nnz);
    assert(find_csr);
    torch::Tensor mask = at::unsqueeze(at::not_equal(compressed_values, 0), -1).repeat({1, 1, block_size[0] * block_size[1]}) ;
    sparse_tensor = at::masked_select(reshaped_tensor, mask);
    index_tensors.push_back(row_indices);
    index_tensors.push_back(compressed_col_idx);
}

