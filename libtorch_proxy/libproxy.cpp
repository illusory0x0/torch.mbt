#include <torch/torch.h>
#include <iostream>

int for_poc_internal() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  return 0;
}