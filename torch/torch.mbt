test "random" {
  let random_6_floats = for_poc_ffi() |> bytes_to_float_array
  for float in random_6_floats {
    println(float)
    assert_true!(float > 0.0)
    assert_true!(float < 1.0)
  }
}

test "tensor" {
  let tensor_a = tensor_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tensor_b = tensor_from_array([1, 2, 3])
  println(tensor_a)
  println(tensor_b)
}