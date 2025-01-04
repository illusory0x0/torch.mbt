# tch_mbt

A PoC of using libtorch in MoonBit.

Will try to make it a library, but for [some reason](https://github.com/moonbitlang/x/issues/70#issuecomment-2469536770):

> ... is not ready for external usage for now due to some native link flag issue.

It may not be as soon as expected.

## Installation

Currently, only Linux is supported.

- Clone this repo.
- Install libtorch (<https://pytorch.org/cppdocs/installing.html>) and CMake (and possibly `build-essential`).
- Edit `build.sh` to set the correct path of libtorch.
- Run `bash build.sh`.

This will eventually build the shared library `libtchproxy.so` and run a simple test (assert 6 random floats is in 0.0 ~ 1.0) by `moon test --target native`.

```
bash build.sh
-- Configuring done
-- Generating done
-- Build files have been written to: /home/undef/moonbit/tch-mbt/libtorch_proxy/build
Consolidate compiler generated dependencies of target tchproxy
[100%] Built target tchproxy
0.11847960948944092
0.044190943241119385
0.726997435092926
0.8324601054191589
0.1506635546684265
0.02712196111679077
Total tests: 1, passed: 1, failed: 0.
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

## TODO

- [ ] Build an real inference model demo.
- [ ] Enhance build experience.
- [ ] Support more APIs and make them public.

## License

Apache 2.0