test "add" {
  let random_6_floats_bytes = tch_for_poc().to_array()
  for i in 0..=5 {
    let i = i * 4
    let float_int = random_6_floats_bytes[i].to_uint() +
      (random_6_floats_bytes[i + 1].to_uint() << 8) +
      (random_6_floats_bytes[i + 2].to_uint() << 16) +
      (random_6_floats_bytes[i + 3].to_uint() << 24)
    let float = float_int.reinterpret_as_float()
    println(float)
  }
  assert_eq!(0, 0)
}
