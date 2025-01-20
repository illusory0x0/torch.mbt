test "random" {
  let random_6_floats = for_poc_ffi() |> bytes_to_float_array
  for float in random_6_floats {
    assert_true!(float > 0.0)
    assert_true!(float < 1.0)
  }
}

test "tensor" {
  let tensor_a = tensor_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  inspect!(tensor_a, content="Tensor([1, 2, 3, 4, 5, 6])")
  let tensor_b = tensor_from_array([1, 2, 3])
  inspect!(tensor_b, content="Tensor([1, 2, 3])")
}

test "tensor_add" {
  let tensor_a = tensor_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tensor_b = tensor_from_array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
  let tensor_c = tensor_a + tensor_b
  inspect!(tensor_c, content="Tensor([7, 7, 7, 7, 7, 7])")
}
