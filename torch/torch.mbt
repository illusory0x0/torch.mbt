test "demo" {
  let tensor_a = tensor_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tensor_b = tensor_from_array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
  let reshape_b = tensor_b.reshape([6, 1])
  let tensor_c = tensor_a.matmul(reshape_b)
  inspect!(tensor_c, content="Tensor([56])")
  tensor_a.drop()
  tensor_b.drop()
  reshape_b.drop()
  tensor_c.drop()
  let tensor_a = tensor_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tensor_b = -tensor_a
  inspect!(tensor_b, content="Tensor([-1, -2, -3, -4, -5, -6])")
}
