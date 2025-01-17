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

fn float_array_to_bytes(floats: Array[Float]) -> Bytes {
  let n = floats.length()
  let bytes_arr: FixedArray[Byte] = FixedArray::make(n * 4, 0)
  for i in 0..<n {
    let float_int_bytes = floats[i].reinterpret_as_int().to_le_bytes()
    for j in 0..<4 {
      bytes_arr[i * 4 + j] = float_int_bytes[j]
    }
  }
  Bytes::from_fixedarray(bytes_arr)
}

fn double_array_to_bytes(doubles: Array[Double]) -> Bytes {
  let n = doubles.length()
  let bytes_arr: FixedArray[Byte] = FixedArray::make(n * 8, 0)
  for i in 0..<n {
    let double_int_bytes = doubles[i].reinterpret_as_uint64().to_le_bytes()
    for j in 0..<8 {
      bytes_arr[i * 8 + j] = double_int_bytes[j]
    }
  }
  Bytes::from_fixedarray(bytes_arr)
}

fn int_array_to_bytes(ints: Array[Int]) -> Bytes {
  let n = ints.length()
  let bytes_arr: FixedArray[Byte] = FixedArray::make(n * 4, 0)
  for i in 0..<n {
    let int_bytes = ints[i].to_le_bytes()
    for j in 0..<4 {
      bytes_arr[i * 4 + j] = int_bytes[j]
    }
  }
  Bytes::from_fixedarray(bytes_arr)
}

fn int_16_array_to_bytes(ints: Array[Int16]) -> Bytes {
  let n = ints.length()
  let bytes_arr: FixedArray[Byte] = FixedArray::make(n * 2, 0)
  for i in 0..<n {
    bytes_arr[i * 2] = (ints[i] & 0xff).to_byte()
    bytes_arr[i * 2 + 1] = ((ints[i] >> 8) & 0xff).to_byte()
  }
  Bytes::from_fixedarray(bytes_arr)
}

fn int_64_array_to_bytes(ints: Array[Int64]) -> Bytes {
  let n = ints.length()
  let bytes_arr: FixedArray[Byte] = FixedArray::make(n * 8, 0)
  for i in 0..<n {
    let int_bytes = ints[i].to_le_bytes()
    for j in 0..<8 {
      bytes_arr[i * 8 + j] = int_bytes[j]
    }
  }
  Bytes::from_fixedarray(bytes_arr)
}