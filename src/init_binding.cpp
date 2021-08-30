#pragma once

#include "sparse_linear.h"
#include "sparse_tensor.h"
#include "sparse_util.h"

namespace py = pybind11;

void declare_sparse_tensor(py::module &m){
    using Class = SparseTensor;
    std::string pyclass_name = std::string("SparseTensor");
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
       .def(py::init<const std::string &, torch::Tensor &, int &, std::vector<int> & >())
       .def("setName",   &Class::setName)
       .def("getName",   &Class::getName)
       .def_readwrite("name", &Class::name)
       .def("getTensor", &Class::getTensor)
       .def_readwrite("sparse_tensor", &Class::sparse_tensor)
       .def_readwrite("shape", &Class::_shape)
       .def_readwrite("block_size", &Class::_block_size)
       .def("__repr__",
           [](const Class &a) {
               return "<SparseTensor named '" + a.name + "'>";
              }
        );
};

PYBIND11_MODULE (_C, m)
{
    m.def ("mlp_sparse_forward", &SparseLinearOp::forward, "sparse linear forward");
    declare_sparse_tensor(m);
}
