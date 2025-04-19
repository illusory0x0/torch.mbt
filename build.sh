#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
rm -rf target

# build libtorch_proxy
cd libtorch_proxy
mkdir -p build
cd build
cmake ..
make -j
cp libtchproxy.so "$HOME/.moon/lib"

# test
cd "$SCRIPT_DIR"
moon test --target native
