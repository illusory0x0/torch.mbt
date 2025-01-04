fn bytes_to_float_array(bytes: Bytes) -> Array[Float] {
  let n = bytes.length() / 4
  let floats: Array[Float] = Array::new(capacity=n)
  for i in 0..<n {
    let i = i * 4
    let float_int = bytes[i].to_uint() +
      (bytes[i + 1].to_uint() << 8) +
      (bytes[i + 2].to_uint() << 16) +
      (bytes[i + 3].to_uint() << 24)
    floats.push(float_int.reinterpret_as_float())
  }
  floats
}

test "random" {
  let random_6_floats = tch_for_poc() |> bytes_to_float_array
  for float in random_6_floats {
    println(float)
    assert_true!(float > 0.0)
    assert_true!(float < 1.0)
  }
}
