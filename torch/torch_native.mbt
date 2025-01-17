///|
fn tch_for_poc() -> Bytes {
  return for_poc_ffi()
}

///|
extern "C" fn for_poc_ffi() -> Bytes = "for_poc"

///|
extern "C" fn at_tensor_of_data_ffi(data_ptr: Bytes, dims: Bytes, ndims: UInt, element_size_in_bytes: UInt, _type: Int) -> Int = "at_tensor_of_data"
