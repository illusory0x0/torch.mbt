///|
extern "C" fn for_poc_ffi() -> Bytes = "for_poc"

///|
extern "C" fn at_tensor_of_data_ffi(
  data_ptr : Bytes,
  dims : Bytes,
  ndims : UInt,
  element_size_in_bytes : UInt,
  _type : Int
) -> Int = "at_tensor_of_data"

///|
extern "C" fn get_tensor_raw_ffi(tensor_id : Int) -> Bytes = "get_tensor_raw"

///|
extern "C" fn add_tensors_ffi(tensor_id_a : Int, tensor_id_b : Int) -> Int = "add_tensors"
