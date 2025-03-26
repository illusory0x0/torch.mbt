#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <variant>
#include <vector>

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

extern "C" {
#include "moonbit.h"
struct tensor_object_internal {
    torch::Tensor object;
};

void tensor_object_internal_delete(void *object) {
    struct tensor_object_internal *obj =
        (struct tensor_object_internal *)object;
    obj->object.~Tensor();
}

struct tensor_object_internal *
tensor_object_internal_new(torch::Tensor object) {
    struct tensor_object_internal *obj =
        (struct tensor_object_internal *)moonbit_make_external_object(
            tensor_object_internal_delete,
            sizeof(struct tensor_object_internal));
    new (&obj->object) torch::Tensor(object);
    return obj;
}

// https://github.com/pytorch/pytorch/issues/20356#issuecomment-1061667333
// f**k libtorch
tensor_object_internal *load_tensor_from_file_internal(moonbit_bytes_t path) {
    try {
        std::vector<char> f = get_the_bytes((char *)path);
        moonbit_decref(path);
        torch::IValue x = torch::pickle_load(f);
        torch::Tensor tensor = x.toTensor();
        return tensor_object_internal_new(tensor);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the tensor\n"
                  << e.what() << "\n"
                  << path << std::endl;
        moonbit_decref(path);
        return nullptr;
    }
}

tensor_object_internal *at_tensor_of_data_internal(moonbit_bytes_t vs,
                                                   moonbit_bytes_t dims,
                                                   size_t ndims,
                                                   size_t element_size_in_bytes,
                                                   int type) {
    auto dims_ = (int64_t *)dims;
    torch::Tensor tensor =
        torch::zeros(torch::IntArrayRef(dims_, ndims), torch::ScalarType(type));
    if ((int64_t)element_size_in_bytes != tensor.element_size())
        throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    moonbit_decref(vs);
    moonbit_decref(dims);
    return tensor_object_internal_new(tensor);
}

tensor_object_internal *reshape_internal(tensor_object_internal *tensor,
                                         moonbit_bytes_t dims, size_t ndims) {
    auto result =
        tensor->object.reshape(torch::IntArrayRef((int64_t *)dims, ndims));
    moonbit_decref(dims);
    return tensor_object_internal_new(result);
}

moonbit_bytes_t get_tensor_raw_internal(tensor_object_internal *tensor) {
    torch::Tensor contiguous_tensor = tensor->object.contiguous();
    void *tensor_data = contiguous_tensor.data_ptr();
    auto size = contiguous_tensor.numel() * contiguous_tensor.element_size();
    moonbit_bytes_t bytes = moonbit_make_bytes(size, 0);
    memcpy(bytes, tensor_data, size);
    return bytes;
}

int get_tensor_length_internal(tensor_object_internal *tensor) {
    return tensor->object.numel();
}

moonbit_bytes_t get_tensor_shape_internal(tensor_object_internal *tensor) {
    std::vector<unsigned> vec(tensor->object.sizes().begin(),
                              tensor->object.sizes().end());
    moonbit_bytes_t bytes =
        moonbit_make_bytes(vec.size() * sizeof(unsigned), 0);
    memcpy(bytes, vec.data(), vec.size() * sizeof(unsigned));
    return bytes;
}

tensor_object_internal *abs_tensor_internal(tensor_object_internal *tensor) {
    torch::Tensor result = torch::abs(tensor->object);
    return tensor_object_internal_new(result);
}

tensor_object_internal *exp_tensor_internal(tensor_object_internal *tensor) {
    torch::Tensor result = torch::exp(tensor->object);
    return tensor_object_internal_new(result);
}

tensor_object_internal *log_tensor_internal(tensor_object_internal *tensor) {
    torch::Tensor result = torch::log(tensor->object);
    return tensor_object_internal_new(result);
}

tensor_object_internal *add_tensors_internal(tensor_object_internal *tensor1,
                                             tensor_object_internal *tensor2) {
    torch::Tensor result = tensor1->object + tensor2->object;
    return tensor_object_internal_new(result);
}

tensor_object_internal *neg_tensor_internal(tensor_object_internal *tensor) {
    torch::Tensor result = -tensor->object;
    return tensor_object_internal_new(result);
}

tensor_object_internal *sub_tensors_internal(tensor_object_internal *tensor1,
                                             tensor_object_internal *tensor2) {
    torch::Tensor result = tensor1->object - tensor2->object;
    return tensor_object_internal_new(result);
}

int equal_tensors_internal(tensor_object_internal *tensor1,
                           tensor_object_internal *tensor2) {
    return torch::allclose(tensor1->object, tensor2->object);
}

tensor_object_internal *mul_tensors_internal(tensor_object_internal *tensor1,
                                             tensor_object_internal *tensor2) {
    torch::Tensor result = tensor1->object * tensor2->object;
    return tensor_object_internal_new(result);
}

tensor_object_internal *
matmul_tensors_internal(tensor_object_internal *tensor1,
                        tensor_object_internal *tensor2) {
    torch::Tensor result = torch::matmul(tensor1->object, tensor2->object);
    return tensor_object_internal_new(result);
}

tensor_object_internal *
transpose_tensor_internal(tensor_object_internal *tensor) {
    torch::Tensor result = tensor->object.t();
    return tensor_object_internal_new(result);
}

tensor_object_internal *argmin_tensor_internal(tensor_object_internal *tensor) {
    torch::Tensor result = torch::argmin(tensor->object);
    return tensor_object_internal_new(result);
}

tensor_object_internal *argmax_tensor_internal(tensor_object_internal *tensor) {
    torch::Tensor result = torch::argmax(tensor->object);
    return tensor_object_internal_new(result);
}

struct module_object_internal {
    torch::jit::script::Module object;
};

void module_object_internal_delete(void *object) {
    struct module_object_internal *obj =
        (struct module_object_internal *)object;
    obj->object.~Module();
}

struct module_object_internal *
module_object_internal_new(torch::jit::script::Module object) {
    struct module_object_internal *obj =
        (struct module_object_internal *)moonbit_make_external_object(
            module_object_internal_delete,
            sizeof(struct module_object_internal));
    new (&obj->object) torch::jit::script::Module(object);
    return obj;
}

module_object_internal *load_model_internal(moonbit_bytes_t path) {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load((char *)path);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n"
                  << e.what() << "\n"
                  << path << std::endl;
        moonbit_decref(path);
        return nullptr;
    }
    moonbit_decref(path);
    return module_object_internal_new(module);
}

tensor_object_internal *forward_internal(module_object_internal *module,
                                         tensor_object_internal *tensor) {
    torch::Tensor result =
        module->object.forward({tensor->object}).toTensor();
    return tensor_object_internal_new(result);
}
}