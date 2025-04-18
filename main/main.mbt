///|
fn main {
  let model = @torch.load_model_from_file("python_examples/mnist/mnist_cnn.pt")
  let input_shape = [1, 1, 28, 28]
  let cases = ["1", "2", "3", "4", "5"]
  let expected_answers = [7, 2, 1, 0, 4]
  for i in 0..<5 {
    let filename = "python_examples/mnist/samples/mnist_" + cases[i] + ".pt"
    let input : @torch.Tensor[Float] = @torch.tensor_from_file(filename).reshape(
      input_shape,
    )
    let output : @torch.Tensor[Float] = model.forward(input)
    let output = output.argmax().get_raw_data()[0]
    println(
      "Expected: " +
      expected_answers[i].to_string() +
      ", Output: " +
      output.to_string(),
    )
  }
}
