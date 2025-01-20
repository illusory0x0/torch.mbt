#include "moonbit.hpp"

#include <cstring>
#include <iostream>
#include <vector>

using tensor_id = int64_t;

std::vector<float> for_poc_internal();

struct moonbit_bytes *for_poc() {
    auto internal_result = for_poc_internal();
    auto size = internal_result.size() * sizeof(float);
    struct moonbit_bytes *bytes = moonbit_make_bytes(size, 0);
    for (int i = 0; i < internal_result.size(); i++) {
        memcpy(bytes->data + i * sizeof(float), &internal_result[i],
               sizeof(float));
    }
    return bytes;
}

tensor_id at_tensor_of_data_internal(void *vs, int64_t *dims, size_t ndims,
                                     size_t element_size_in_bytes, int type);

tensor_id at_tensor_of_data(struct moonbit_bytes *data_ptr,
                            struct moonbit_bytes *dims, unsigned ndims,
                            unsigned element_size_in_bytes, int type) {
    return at_tensor_of_data_internal(data_ptr->data, (int64_t *)dims->data,
                                      ndims, element_size_in_bytes, type);
}

std::vector<unsigned char> get_tensor_raw_internal(tensor_id global_id);

struct moonbit_bytes *get_tensor_raw(tensor_id global_id) {
    auto internal_result = get_tensor_raw_internal(global_id);
    auto size = internal_result.size();
    struct moonbit_bytes *bytes = moonbit_make_bytes(size, 0);
    for (int i = 0; i < internal_result.size(); i++) {
        bytes->data[i] = internal_result[i];
    }
    return bytes;
}

void drop_tensor_internal(tensor_id global_id);

tensor_id add_tensors_internal(tensor_id global_id1, tensor_id global_id2);

tensor_id neg_tensor_internal(tensor_id global_id);

tensor_id sub_tensors_internal(tensor_id global_id1, tensor_id global_id2);

int equal_tensors_internal(tensor_id global_id1, tensor_id global_id2);

tensor_id mul_tensors_internal(tensor_id global_id1, tensor_id global_id2);

tensor_id matmul_tensors_internal(tensor_id global_id1, tensor_id global_id2);

tensor_id reshape_internal(tensor_id global_id, int64_t *dims, size_t ndims);

tensor_id reshape(tensor_id global_id, struct moonbit_bytes *dims,
                  unsigned ndims) {
    return reshape_internal(global_id, (int64_t *)dims->data, ndims);
}

std::vector<unsigned> get_tensor_shape_internal(tensor_id global_id);

struct moonbit_bytes *get_tensor_shape(tensor_id global_id) {
    auto internal_result = get_tensor_shape_internal(global_id);
    auto size = internal_result.size() * sizeof(unsigned);
    struct moonbit_bytes *bytes = moonbit_make_bytes(size, 0);
    for (int i = 0; i < internal_result.size(); i++) {
        memcpy(bytes->data + i * sizeof(unsigned), &internal_result[i],
               sizeof(unsigned));
    }
    return bytes;
}