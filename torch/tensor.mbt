enum ElementKind {
    Uint8
    Int8
    Int16
    Int
    Int64
    Half
    Float
    Double
    ComplexHalf
    ComplexFloat
    ComplexDouble
    Bool
    QInt8
    QUInt8
    QInt32
    BFloat16
    Float8e5m2
    Float8e4m3fn
    Float8e5m2fnuz
    Float8e4m3fnuz
}

trait Element {
  kind(Self) -> ElementKind
  elt_size_in_bytes(Self) -> UInt
  c_int(Self) -> Int
  bytes(Self, Array[Self]) -> Bytes
}

impl Element with elt_size_in_bytes(self) {
  match self.kind() {
    ElementKind::Uint8 => 1
    ElementKind::Int8 => 1
    ElementKind::Int16 => 2
    ElementKind::Int => 4
    ElementKind::Int64 => 8
    ElementKind::Half => 2
    ElementKind::Float => 4
    ElementKind::Double => 8
    ElementKind::ComplexHalf => 4
    ElementKind::ComplexFloat => 8
    ElementKind::ComplexDouble => 16
    ElementKind::Bool => 1
    ElementKind::QInt8 => 1
    ElementKind::QUInt8 => 1
    ElementKind::QInt32 => 4
    ElementKind::BFloat16 => 2
    ElementKind::Float8e5m2 => 1
    ElementKind::Float8e4m3fn => 1
    ElementKind::Float8e5m2fnuz => 1
    ElementKind::Float8e4m3fnuz => 1
  }
}


// These values should be in sync with include/c10/core/ScalarType.h
impl Element with c_int(self) {
  match self.kind() {
    ElementKind::Uint8 => 0
    ElementKind::Int8 => 1
    ElementKind::Int16 => 2
    ElementKind::Int => 3
    ElementKind::Int64 => 4
    ElementKind::Half => 5
    ElementKind::Float => 6
    ElementKind::Double => 7
    ElementKind::ComplexHalf => 8
    ElementKind::ComplexFloat => 9
    ElementKind::ComplexDouble => 10
    ElementKind::Bool => 11
    ElementKind::QInt8 => 12
    ElementKind::QUInt8 => 13
    ElementKind::QInt32 => 14
    ElementKind::BFloat16 => 15
    ElementKind::Float8e5m2 => 23
    ElementKind::Float8e4m3fn => 24
    ElementKind::Float8e5m2fnuz => 25
    ElementKind::Float8e4m3fnuz => 26
  }
}

impl Element for Int16 with kind(_) { ElementKind::Int16 }
impl Element for Int16 with bytes(_, arr) { 
  int_16_array_to_bytes(arr)
}
impl Element for Int with kind(_) { ElementKind::Int }
impl Element for Int with bytes(_, arr) { 
  int_array_to_bytes(arr)
}
impl Element for Int64 with kind(_) { ElementKind::Int64 }
impl Element for Int64 with bytes(_, arr) { 
  int_64_array_to_bytes(arr)
}
impl Element for Float with kind(_) { ElementKind::Float }
impl Element for Float with bytes(_, arr) { 
  float_array_to_bytes(arr)
}
impl Element for Double with kind(_) { ElementKind::Double }
impl Element for Double with bytes(_, arr) { 
  double_array_to_bytes(arr)
}

struct Tensor {
  id: Int
}

impl Show for Tensor with output(self, logger) {
  logger.write_string("Tensor(")
  logger.write_string(self.id.to_string())
  logger.write_string(")")
}

fn at_tensor_of_data(data_ptr: Bytes, dims: Array[Int], ndims: UInt, element_size_in_bytes: UInt, _type: Int) -> Tensor {
  let dims = int_array_to_bytes(dims)
  let id = at_tensor_of_data_ffi(data_ptr, dims, ndims, element_size_in_bytes, _type)
  { id: id }
}
fn tensor_from_array[T: Element](arr: Array[T]) -> Tensor {
  let raw = arr[0].bytes(arr)
  at_tensor_of_data(raw, [arr.length()], 1, arr[0].elt_size_in_bytes(), arr[0].c_int())
}