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
}
```

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