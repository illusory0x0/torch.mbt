#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <torch/torch.h>
#include <vector>

using tensor_id = int64_t;

std::atomic<tensor_id> next_id{0};

tensor_id create_new_id() {
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

std::map<tensor_id, torch::Tensor> global_tensor_map;

tensor_id insert_new_tensor(torch::Tensor tensor) {
    tensor_id new_id = create_new_id();
    global_tensor_map.insert({new_id, std::move(tensor)});
    return new_id;
}

extern "C" {
tensor_id at_tensor_of_data_internal(void *vs, int64_t *dims, size_t ndims,
                                     size_t element_size_in_bytes, int type) {
    torch::Tensor tensor =
        torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if ((int64_t)element_size_in_bytes != tensor.element_size())
        throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    return insert_new_tensor(std::move(tensor));
}

tensor_id reshape_internal(tensor_id global_id, int64_t *dims, size_t ndims) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    auto result = tensor.reshape(torch::IntArrayRef(dims, ndims));
    return insert_new_tensor(std::move(result));
}

int get_tensor_raw_internal(tensor_id global_id, unsigned char **data) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    torch::Tensor contiguous_tensor = tensor.contiguous();
    void *tensor_data = contiguous_tensor.data_ptr();
    std::vector<unsigned char> vec(contiguous_tensor.numel() *
                                   contiguous_tensor.element_size());
    *data = (unsigned char *)malloc(vec.size());
    memcpy(*data, tensor_data, vec.size());
    return vec.size();
}

int get_tensor_shape_internal(tensor_id global_id, unsigned **shape) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    std::vector<unsigned> vec(tensor.sizes().begin(), tensor.sizes().end());
    *shape = (unsigned *)malloc(vec.size() * sizeof(unsigned));
    memcpy(*shape, vec.data(), vec.size() * sizeof(unsigned));
    return vec.size();
}

void drop_tensor_internal(tensor_id global_id) {
    global_tensor_map.erase(global_id);
}

tensor_id add_tensors_internal(tensor_id global_id1, tensor_id global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    torch::Tensor result = tensor1 + tensor2;
    return insert_new_tensor(std::move(result));
}

tensor_id neg_tensor_internal(tensor_id global_id) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    torch::Tensor result = -tensor;
    return insert_new_tensor(std::move(result));
}

tensor_id sub_tensors_internal(tensor_id global_id1, tensor_id global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    torch::Tensor result = tensor1 - tensor2;
    return insert_new_tensor(std::move(result));
}

int equal_tensors_internal(tensor_id global_id1, tensor_id global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    return torch::allclose(tensor1, tensor2);
}

tensor_id mul_tensors_internal(tensor_id global_id1, tensor_id global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    torch::Tensor result = tensor1 * tensor2;
    return insert_new_tensor(std::move(result));
}

tensor_id matmul_tensors_internal(tensor_id global_id1, tensor_id global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    torch::Tensor result = torch::matmul(tensor1, tensor2);
    return insert_new_tensor(std::move(result));
}

tensor_id transpose_tensor_internal(tensor_id global_id) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    torch::Tensor result = tensor.t();
    return insert_new_tensor(std::move(result));
}
}