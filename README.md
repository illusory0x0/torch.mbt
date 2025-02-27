# tch_mbt

Using libtorch in MoonBit.

For [some native link reason](https://github.com/moonbitlang/x/issues/70#issuecomment-2469536770), it's now a starter template instead of a library.

## Setup

Only Linux is supported.

- Clone the repo.
- Install libtorch (<https://pytorch.org/cppdocs/installing.html>).
- Install CMake (and possibly "build-essential").
- Edit `build.sh` to set the correct path of libtorch.
- Run `bash build.sh`.

You are expected to see the unittests passed.

## Usage

Change code in `torch/torch.mbt` and run `bash build.sh` to test.

```moonbit
// torch/torch.mbt
// Load a model and do inference on MNIST dataset.
// You can check the images in python_examples/mnist/samples.
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

Check the full list in [torch.mbti](torch/torch.mbti).

## How it works

tch_mbt uses the default cc to compile the native backend of MoonBit. However, libtorch is a C++ library and requires a build system like CMake. To bridge the gap, tch_mbt will first build a shared "C" library "tchproxy".

```json
{
    "link": {
        "native": {
            "cc-flags": "-L. -ltchproxy"
        }
    },
    "native-stub": [
        "native_stub.c"
    ]
}
```

`native_stub.c` is here for interoperability, wrapping FFIs that either the arguments or the return value is a MoonBit object.

## Roadmap & TODOs

- [x] Basic tensor operations.
- [x] Basic neural network forward pass.
- [x] Build an real inference model demo.
- [ ] Add more tensor operations.
- [ ] Add more neural network operations.
- [ ] Enhance build experience.

## License

Apache 2.0