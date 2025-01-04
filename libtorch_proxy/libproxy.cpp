#include <torch/torch.h>
#include <iostream>
#include <vector>

std::vector<float> for_poc_internal() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::vector<float> vec(tensor.data_ptr<float>(),
                         tensor.data_ptr<float>() + tensor.numel());
  return vec;
}