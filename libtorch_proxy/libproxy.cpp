#include <iostream>
#include <map>
#include <torch/torch.h>
#include <vector>

std::vector<float> for_poc_internal() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::vector<float> vec(tensor.data_ptr<float>(),
                           tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

std::atomic<int64_t> next_id{0};

int create_new_id() { return next_id.fetch_add(1, std::memory_order_relaxed); }

std::map<int64_t, torch::Tensor> global_tensor_map;

int64_t insert_new_tensor(torch::Tensor tensor) {
    int64_t new_id = create_new_id();
    global_tensor_map.insert({new_id, std::move(tensor)});
    return new_id;
}

int64_t at_tensor_of_data_internal(void *vs, int64_t *dims, size_t ndims,
                                   size_t element_size_in_bytes, int type) {
    torch::Tensor tensor =
        torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if ((int64_t)element_size_in_bytes != tensor.element_size())
        throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    return insert_new_tensor(std::move(tensor));
}

int64_t reshape_internal(int64_t global_id, int64_t *dims, size_t ndims) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    auto result = tensor.reshape(torch::IntArrayRef(dims, ndims));
    return insert_new_tensor(std::move(result));
}

std::vector<unsigned char> get_tensor_raw_internal(int64_t global_id) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    void *tensor_data = tensor.data_ptr();
    std::vector<unsigned char> vec(tensor.numel() * tensor.element_size());
    memcpy(vec.data(), tensor_data, vec.size());
    return vec;
}

std::vector<unsigned> get_tensor_shape_internal(int64_t global_id) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    std::vector<unsigned> vec(tensor.sizes().begin(), tensor.sizes().end());
    return vec;
}

void drop_tensor_internal(int64_t global_id) {
    global_tensor_map.erase(global_id);
}

int64_t add_tensors_internal(int64_t global_id1, int64_t global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    torch::Tensor result = tensor1 + tensor2;
    return insert_new_tensor(std::move(result));
}

int64_t neg_tensor_internal(int64_t global_id) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    torch::Tensor result = -tensor;
    return insert_new_tensor(std::move(result));
}

int64_t sub_tensors_internal(int64_t global_id1, int64_t global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    torch::Tensor result = tensor1 - tensor2;
    return insert_new_tensor(std::move(result));
}

int equal_tensors_internal(int64_t global_id1, int64_t global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    return torch::allclose(tensor1, tensor2);
}

int64_t mul_tensors_internal(int64_t global_id1, int64_t global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    torch::Tensor result = tensor1 * tensor2;
    return insert_new_tensor(std::move(result));
}

int64_t matmul_tensors_internal(int64_t global_id1, int64_t global_id2) {
    torch::Tensor &tensor1 = global_tensor_map[global_id1];
    torch::Tensor &tensor2 = global_tensor_map[global_id2];
    torch::Tensor result = torch::matmul(tensor1, tensor2);
    return insert_new_tensor(std::move(result));
}
