#pragma once

#include <torch/extension.h>
#include <vector>
#include <assert.h>
#include <string.h>
#include "sparse_util.h"

struct SparseTensor{
    SparseTensor(const std::string &name,
                 torch::Tensor dense_tensor,
                 int format,
		 std::vector<int> block_size_) :
                name(name){
                block_size[0] = block_size_[0];
                block_size[1] = block_size_[1];
                shape[0] = dense_tensor.size(0);
                shape[1] = dense_tensor.size(1);
                assert(format ==0 || format==1);
                _shape.assign(shape, shape+2);
                _block_size = block_size_;
                format = SparseCompression(format);
		compress(dense_tensor);
            }

    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }
    torch::Tensor &getTensor() {return sparse_tensor; }
    void setTensor(torch::Tensor tensor) { compress(tensor.contiguous()); }
    float getSparsity(){
        return sparse_tensor.numel() / ( shape[0] * shape[1] );
    }

    void compress(torch::Tensor dense_tensor){
        if (int(format) == 0){
            create_bsr_tensors(dense_tensor, sparse_tensor, index_tensors, block_size, shape);
	}else{
            create_bsc_tensors(dense_tensor, sparse_tensor, index_tensors, block_size, shape);
        }
        printf("compressed done \n");
    };

    torch::Tensor sparse_tensor;
    std::vector<torch::Tensor> index_tensors;
    std::string name;
    SparseCompression format;
    int shape[2] = {0, 0};
    int block_size[2] = {1, 1};
    std::vector<int> _shape;
    std::vector<int> _block_size;
};
