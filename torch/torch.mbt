///|
test "demo" {
  let tensor_a = tensor_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tensor_b = tensor_from_array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
  let reshape_b = tensor_b.reshape([6, 1])
  let tensor_c = tensor_a.matmul(reshape_b)
  inspect!(tensor_c, content="Tensor([56])")
  tensor_a.drop()
  tensor_b.drop()
  reshape_b.drop()
  tensor_c.drop()
  let tensor_a = tensor_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tensor_b = -tensor_a
  inspect!(tensor_b, content="Tensor([-1, -2, -3, -4, -5, -6])")
}

///|
test "inference" {
  let model = load_model_from_file("python_examples/mnist/mnist_cnn.pt")
  let cases = ["1", "2", "3", "4", "5"]
  let expected_answers = [7, 2, 1, 0, 4]
  fn argmax(v : Array[Float]) -> Int {
    v.foldi(init=0, fn(i, acc, x) { if x > v[acc] { i } else { acc } })
  }

  for i in 0..<5 {
    let filename = "python_examples/mnist/samples/mnist_" + cases[i] + ".pt"
    let input : Tensor[Float] = tensor_from_file(filename)
    let input_resized = input.reshape([1, 1, 28, 28])
    let output : Tensor[Float] = model.forward(input_resized)
    let output_vec = output.get_raw_data()
    let index_max = argmax(output_vec)
    assert_eq!(index_max, expected_answers[i])
    input.drop()
    input_resized.drop()
    output.drop()
  }
  model.drop()
}
