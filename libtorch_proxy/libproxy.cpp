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

std::map<int, torch::Tensor> global_tensor_map;

int at_tensor_of_data_internal(void *vs, int64_t *dims, size_t ndims,
                               size_t element_size_in_bytes, int type) {
    torch::Tensor tensor =
        torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if ((int64_t)element_size_in_bytes != tensor.element_size())
        throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    global_tensor_map.insert(
        std::pair<int, torch::Tensor>(global_tensor_map.size(), std::move(tensor)));
    return global_tensor_map.size() - 1;
}

std::vector<unsigned char> get_tensor_raw_internal(int global_id) {
    torch::Tensor &tensor = global_tensor_map[global_id];
    void *tensor_data = tensor.data_ptr();
    std::vector<unsigned char> vec(tensor.numel() * tensor.element_size());
    memcpy(vec.data(), tensor_data, vec.size());
    return vec;
}

// void drop_tensor(int global_id) { global_tensor_map.erase(global_id); }

// int multiply_tensors(int global_id1, int global_id2) {
//     torch::Tensor& tensor1 = global_tensor_map[global_id1];
//     torch::Tensor& tensor2 = global_tensor_map[global_id2];
//     torch::Tensor result = tensor1 * tensor2;
//     global_tensor_map.insert(
//         std::pair<int, torch::Tensor>(global_tensor_map.size(), result));
//     return global_tensor_map.size() - 1;
// }
