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
extern "C" fn drop_tensor_ffi(tensor_id : Int) = "drop_tensor_internal"

///|
extern "C" fn add_tensors_ffi(tensor_id_a : Int, tensor_id_b : Int) -> Int = "add_tensors_internal"

///|
extern "C" fn neg_tensor_ffi(tensor_id : Int) -> Int = "neg_tensor_internal"

///|
extern "C" fn sub_tensors_ffi(tensor_id_a : Int, tensor_id_b : Int) -> Int = "sub_tensors_internal"

///|
extern "C" fn equal_tensors_ffi(tensor_id_a : Int, tensor_id_b : Int) -> Int = "equal_tensors_internal"

///|
extern "C" fn mul_tensors_ffi(tensor_id_a : Int, tensor_id_b : Int) -> Int = "mul_tensors_internal"

///|
extern "C" fn matmul_tensors_ffi(tensor_id_a : Int, tensor_id_b : Int) -> Int = "matmul_tensors_internal"

///|
extern "C" fn reshape_ffi(tensor_id : Int, dims : Bytes, ndims : UInt) -> Int = "reshape"

///|
extern "C" fn get_tensor_shape_ffi(tensor_id : Int) -> Bytes = "get_tensor_shape"
