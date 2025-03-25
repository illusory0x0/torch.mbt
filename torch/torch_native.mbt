///|
type TensorObject

///|
type ModelObject

///|
extern "C" fn at_tensor_of_data_ffi(
  data_ptr : Bytes,
  dims : Bytes,
  ndims : UInt,
  element_size_in_bytes : UInt,
  _type : Int
) -> TensorObject = "at_tensor_of_data_internal"

///|
extern "C" fn load_tensor_from_file_ffi(path : Bytes) -> TensorObject = "load_tensor_from_file_internal"

///|
extern "C" fn get_tensor_raw_ffi(tensor_obj : TensorObject) -> Bytes = "get_tensor_raw_internal"

///|
extern "C" fn argmin_tensor_ffi(tensor_obj : TensorObject) -> TensorObject = "argmin_tensor_internal"

///|
extern "C" fn argmax_tensor_ffi(tensor_obj : TensorObject) -> TensorObject = "argmax_tensor_internal"

///|
extern "C" fn abs_tensor_ffi(tensor_obj : TensorObject) -> TensorObject = "abs_tensor_internal"

///|
extern "C" fn exp_tensor_ffi(tensor_obj : TensorObject) -> TensorObject = "exp_tensor_internal"

///|
extern "C" fn log_tensor_ffi(tensor_obj : TensorObject) -> TensorObject = "log_tensor_internal"

///|
extern "C" fn add_tensors_ffi(
  tensor_obj_a : TensorObject,
  tensor_obj_b : TensorObject
) -> TensorObject = "add_tensors_internal"

///|
extern "C" fn neg_tensor_ffi(tensor_obj : TensorObject) -> TensorObject = "neg_tensor_internal"

///|
extern "C" fn sub_tensors_ffi(
  tensor_obj_a : TensorObject,
  tensor_obj_b : TensorObject
) -> TensorObject = "sub_tensors_internal"

///|
extern "C" fn equal_tensors_ffi(
  tensor_obj_a : TensorObject,
  tensor_obj_b : TensorObject
) -> Int = "equal_tensors_internal"

///|
extern "C" fn mul_tensors_ffi(
  tensor_obj_a : TensorObject,
  tensor_obj_b : TensorObject
) -> TensorObject = "mul_tensors_internal"

///|
extern "C" fn matmul_tensors_ffi(
  tensor_obj_a : TensorObject,
  tensor_obj_b : TensorObject
) -> TensorObject = "matmul_tensors_internal"

///|
extern "C" fn transpose_tensor_ffi(tensor_obj : TensorObject) -> TensorObject = "transpose_tensor_internal"

///|
extern "C" fn reshape_ffi(
  tensor_obj : TensorObject,
  dims : Bytes,
  ndims : UInt
) -> TensorObject = "reshape_internal"

///|
extern "C" fn get_tensor_length_ffi(tensor_obj : TensorObject) -> Int = "get_tensor_length_internal"

///|
extern "C" fn get_tensor_shape_ffi(tensor_obj : TensorObject) -> Bytes = "get_tensor_shape_internal"

///|
extern "C" fn load_model_ffi(path : Bytes) -> ModelObject = "load_model_internal"

///|
extern "C" fn forward_ffi(
  model_object : ModelObject,
  input_object : TensorObject
) -> TensorObject = "forward_internal"
