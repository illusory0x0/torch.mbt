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
