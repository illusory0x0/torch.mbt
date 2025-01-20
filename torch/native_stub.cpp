#include "moonbit.hpp"

#include <cstring>
#include <iostream>
#include <vector>

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

int at_tensor_of_data_internal(void *vs, int64_t *dims, size_t ndims,
                               size_t element_size_in_bytes, int type);

int at_tensor_of_data(struct moonbit_bytes *data_ptr,
                      struct moonbit_bytes *dims, unsigned ndims,
                      unsigned element_size_in_bytes, int type) {
    return at_tensor_of_data_internal(data_ptr->data, (int64_t *)dims->data,
                                      ndims, element_size_in_bytes, type);
}

std::vector<unsigned char> get_tensor_raw_internal(int global_id);

struct moonbit_bytes *get_tensor_raw(int global_id) {
    auto internal_result = get_tensor_raw_internal(global_id);
    auto size = internal_result.size();
    struct moonbit_bytes *bytes = moonbit_make_bytes(size, 0);
    for (int i = 0; i < internal_result.size(); i++) {
        bytes->data[i] = internal_result[i];
    }
    return bytes;
}

void drop_tensor_internal(int global_id);

void drop_tensor(int global_id) { drop_tensor_internal(global_id); }

int add_tensors_internal(int global_id1, int global_id2);

int add_tensors(int global_id1, int global_id2) {
    return add_tensors_internal(global_id1, global_id2);
}