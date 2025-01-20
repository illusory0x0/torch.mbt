test "random" {
  let random_6_floats = for_poc_ffi() |> bytes_to_float_array
  for float in random_6_floats {
    println(float)
    assert_true!(float > 0.0)
    assert_true!(float < 1.0)
  }
}
