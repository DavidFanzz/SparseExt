include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

pybind11_add_module(_C MODULE
    init_binding.cpp
    )

target_link_libraries(_C
    PRIVATE "${TORCH_LIBRARIES}"
    PRIVATE "${TORCH_PYTHON_LIBRARY}"
    )

target_compile_definitions(_C PRIVATE)
