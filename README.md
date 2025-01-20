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
test "tensor_add" {
  let tensor_a = tensor_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tensor_b = tensor_from_array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
  let tensor_c = tensor_a + tensor_b
  inspect!(tensor_c, content="Tensor([7, 7, 7, 7, 7, 7])")
  tensor_a.drop()
  tensor_b.drop()
  tensor_c.drop()
}
```

## How it works

First we need to make MoonBit work with C++. We replace the default `tcc` with `g++` to enable C++ support.

```json
"link": {
   "native": {
      "cc": "g++",
      "cc-flags": "./torch/native_stub.cpp",
      "cc-link-flags": "-I . -L. -ltchproxy -lm"
   }
}
```

To support C++, there is another stuff to do: mark some functions in "moonbit.h" as `extern "C"`. You can see it in "moonbit.hpp".

However, libtorch not only needs C++ compiler support, but also requires a build system like CMake. We offload the CMake build to a separate directory `libtorch_proxy` and then copy the shared library to the MoonBit project, which is the final FFI solution.

> Potential optimization: We can hook g++ with a custom cc script, so we don't need a seperate shared library.

## Roadmap & TODOs

- [x] Basic tensor operations.
- [ ] Basic neural network operations.
- [ ] Build an real inference model demo.
- [ ] Enhance build experience.

## License

Apache 2.0