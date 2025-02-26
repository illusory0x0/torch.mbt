///|
struct Model {
  id : Int64
}

///|
pub fn load_model(path : String) -> Model {
  let path = mbt_string_to_utf8_bytes(path, true)
  let id = load_model_ffi(path)
  { id, }
}

///|
pub fn forward[U, V](self : Model, input : Tensor[U]) -> Tensor[V] {
  let id = forward_ffi(self.id, input.id)
  { id, }
}

///|
pub fn Model::drop(self : Model) -> Unit {
  drop_torch_object(self.id)
}

///|
test "inference" {
  let model = load_model("python_examples/mnist/mnist_cnn.pt")
  let cases = ["1", "2", "3", "4", "5"]
  let results = [7, 2, 1, 0, 4]
  fn get_index_max(v : Array[Float]) -> Int {
    let mut max : Float = (0xFF800000).reinterpret_as_float()
    let mut index_max = 0
    for i in 0..<v.length() {
      if v[i] > max {
        max = v[i]
        index_max = i
      }
    }
    index_max
  }

  for i in 0..<5 {
    let input : Tensor[Float] = tensor_from_file(
      "python_examples/mnist/samples/mnist_" + cases[i] + ".pt",
    )
    let input_resized = input.reshape([1, 1, 28, 28])
    let output : Tensor[Float] = model.forward(input_resized)
    let output_vec = Float::to_vec(get_tensor_raw_ffi(output.id))
    let index_max = get_index_max(output_vec)
    assert_eq!(index_max, results[i])
    input.drop()
    input_resized.drop()
    output.drop()
  }
  model.drop()
}
