#include "moonbit.hpp"

#include <iostream>
#include <vector>
#include <cstring>

std::vector<float> for_poc_internal();

struct moonbit_bytes* for_poc() {
  auto internal_result = for_poc_internal();
  auto size = internal_result.size() * sizeof(float);
  struct moonbit_bytes* bytes = moonbit_make_bytes(size, 0);
  for (int i = 0; i < internal_result.size(); i++) {
    memcpy(bytes->data + i * sizeof(float), &internal_result[i], sizeof(float));
  }
  return bytes;
}