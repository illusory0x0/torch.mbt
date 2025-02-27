# tch_mbt

A PoC of using libtorch in MoonBit.

Will try to make it a library, but for [some reason](https://github.com/moonbitlang/x/issues/70#issuecomment-2469536770):

> ... is not ready for external usage for now due to some native link flag issue.

It may not be as soon as expected.

## Setup

Currently, only Linux is supported.

- Clone this repo.
- Install libtorch (<https://pytorch.org/cppdocs/installing.html>) and CMake (and possibly `build-essential`).
- Edit `build.sh` to set the correct path of libtorch.
- Run `bash build.sh`.

This will eventually build the shared library `libtchproxy.so` and run tests.

## Usage

You may change `torch/torch.mbt` and run `bash build.sh` to test.

```moonbit
test "inference" {
  let model = load_model_from_file("python_examples/mnist/mnist_cnn.pt")
  let cases = ["1", "2", "3", "4", "5"]
  let expected_answers = [7, 2, 1, 0, 4]
  for i in 0..<5 {
    let filename = "python_examples/mnist/samples/mnist_" + cases[i] + ".pt"
    let input : Tensor[Float] = tensor_from_file(filename)
    let input_resized = input.reshape([1, 1, 28, 28])
    let output : Tensor[Float] = model.forward(input_resized)
    let output_argmax = output.argmax()
    let index_max = output_argmax.get_raw_data()[0]
    assert_eq!(index_max, expected_answers[i].to_int64())
    input.drop()
    input_resized.drop()
    output.drop()
    output_argmax.drop()
  }
  model.drop()
}
```

## API

You can check the full list in [torch.mbti](torch/torch.mbti).

## How it works

For some compatibility issues, tch_mbt now uses gcc to compile the native backend of MoonBit. However, libtorch is a C++ library and requires a build system like CMake. To bridge the gap, tch_mbt builds a libtorch shared library and links it with the native backend.

```json
"link": {
   "native": {
      "cc": "gcc",
      "cc-flags": "./torch/native_stub.c",
      "cc-link-flags": "-L. -ltchproxy -lm"
   }
}
```

`native_stub.c` is kept for interoperability, providing APIs where either the arguments or the return value is a MoonBit object.

## Roadmap & TODOs

- [x] Basic tensor operations.
- [x] Basic neural network forward pass.
- [x] Build an real inference model demo.
- [ ] Add more tensor operations.
- [ ] Add more neural network operations.
- [ ] Enhance build experience.

## License

Apache 2.0