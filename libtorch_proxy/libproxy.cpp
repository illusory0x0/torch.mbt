#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <variant>
#include <vector>

using torch_object_id = int64_t;

using torch_object = std::variant<torch::Tensor, torch::jit::script::Module>;

std::atomic<torch_object_id> next_id{0};

torch_object_id create_new_id() {
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

std::unordered_map<torch_object_id, torch_object> global_torch_map;

torch_object_id insert_new_torch_object(torch_object tensor) {
    torch_object_id new_id = create_new_id();
    global_torch_map.insert({new_id, std::move(tensor)});
    return new_id;
}

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

extern "C" {
// https://github.com/pytorch/pytorch/issues/20356#issuecomment-1061667333
// f**k libtorch
torch_object_id load_tensor_from_file_internal(char *path) {
    try {
        std::vector<char> f = get_the_bytes(path);
        torch::IValue x = torch::pickle_load(f);
        torch::Tensor tensor = x.toTensor();
        return insert_new_torch_object(std::move(tensor));
    } catch (const c10::Error &e) {
        std::cerr << "error loading the tensor\n"
                  << e.what() << "\n"
                  << path << std::endl;
        return -1;
    }
}

torch_object_id at_tensor_of_data_internal(void *vs, int64_t *dims,
                                           size_t ndims,
                                           size_t element_size_in_bytes,
                                           int type) {
    torch::Tensor tensor =
        torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if ((int64_t)element_size_in_bytes != tensor.element_size())
        throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    return insert_new_torch_object(std::move(tensor));
}

torch_object_id reshape_internal(torch_object_id global_id, int64_t *dims,
                                 size_t ndims) {
    auto &tensor = std::get<torch::Tensor>(global_torch_map[global_id]);
    auto result = tensor.reshape(torch::IntArrayRef(dims, ndims));
    return insert_new_torch_object(std::move(result));
}

int get_tensor_raw_internal(torch_object_id global_id, unsigned char **data) {
    auto &tensor = std::get<torch::Tensor>(global_torch_map[global_id]);
    torch::Tensor contiguous_tensor = tensor.contiguous();
    void *tensor_data = contiguous_tensor.data_ptr();
    auto size = contiguous_tensor.numel() * contiguous_tensor.element_size();
    *data = (unsigned char *)malloc(size);
    memcpy(*data, tensor_data, size);
    return size;
}

int get_tensor_shape_internal(torch_object_id global_id, unsigned **shape) {
    auto &tensor = std::get<torch::Tensor>(global_torch_map[global_id]);
    std::vector<unsigned> vec(tensor.sizes().begin(), tensor.sizes().end());
    *shape = (unsigned *)malloc(vec.size() * sizeof(unsigned));
    memcpy(*shape, vec.data(), vec.size() * sizeof(unsigned));
    return vec.size();
}

void drop_torch_object_internal(torch_object_id global_id) {
    global_torch_map.erase(global_id);
}

torch_object_id add_tensors_internal(torch_object_id global_id1,
                                     torch_object_id global_id2) {
    auto &tensor1 = std::get<torch::Tensor>(global_torch_map[global_id1]);
    auto &tensor2 = std::get<torch::Tensor>(global_torch_map[global_id2]);
    torch::Tensor result = tensor1 + tensor2;
    return insert_new_torch_object(std::move(result));
}

torch_object_id neg_tensor_internal(torch_object_id global_id) {
    auto &tensor = std::get<torch::Tensor>(global_torch_map[global_id]);
    torch::Tensor result = -tensor;
    return insert_new_torch_object(std::move(result));
}

torch_object_id sub_tensors_internal(torch_object_id global_id1,
                                     torch_object_id global_id2) {
    auto &tensor1 = std::get<torch::Tensor>(global_torch_map[global_id1]);
    auto &tensor2 = std::get<torch::Tensor>(global_torch_map[global_id2]);
    torch::Tensor result = tensor1 - tensor2;
    return insert_new_torch_object(std::move(result));
}

int equal_tensors_internal(torch_object_id global_id1,
                           torch_object_id global_id2) {
    auto &tensor1 = std::get<torch::Tensor>(global_torch_map[global_id1]);
    auto &tensor2 = std::get<torch::Tensor>(global_torch_map[global_id2]);
    return torch::allclose(tensor1, tensor2);
}

torch_object_id mul_tensors_internal(torch_object_id global_id1,
                                     torch_object_id global_id2) {
    auto &tensor1 = std::get<torch::Tensor>(global_torch_map[global_id1]);
    auto &tensor2 = std::get<torch::Tensor>(global_torch_map[global_id2]);
    torch::Tensor result = tensor1 * tensor2;
    return insert_new_torch_object(std::move(result));
}

torch_object_id matmul_tensors_internal(torch_object_id global_id1,
                                        torch_object_id global_id2) {
    auto &tensor1 = std::get<torch::Tensor>(global_torch_map[global_id1]);
    auto &tensor2 = std::get<torch::Tensor>(global_torch_map[global_id2]);
    torch::Tensor result = torch::matmul(tensor1, tensor2);
    return insert_new_torch_object(std::move(result));
}

torch_object_id transpose_tensor_internal(torch_object_id global_id) {
    auto &tensor = std::get<torch::Tensor>(global_torch_map[global_id]);
    torch::Tensor result = tensor.t();
    return insert_new_torch_object(std::move(result));
}

torch_object_id argmin_tensor_internal(torch_object_id global_id) {
    auto &tensor = std::get<torch::Tensor>(global_torch_map[global_id]);
    torch::Tensor result = torch::argmin(tensor);
    return insert_new_torch_object(std::move(result));
}

torch_object_id argmax_tensor_internal(torch_object_id global_id) {
    auto &tensor = std::get<torch::Tensor>(global_torch_map[global_id]);
    torch::Tensor result = torch::argmax(tensor);
    return insert_new_torch_object(std::move(result));
}

torch_object_id load_model_internal(char *path) {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(path);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n"
                  << e.what() << "\n"
                  << path << std::endl;
        return -1;
    }
    return insert_new_torch_object(std::move(module));
}

torch_object_id forward_internal(torch_object_id global_id,
                                 torch_object_id global_id2) {
    auto &module =
        std::get<torch::jit::script::Module>(global_torch_map[global_id]);
    auto &tensor = std::get<torch::Tensor>(global_torch_map[global_id2]);
    torch::Tensor result = module.forward({tensor}).toTensor();
    return insert_new_torch_object(std::move(result));
}
}