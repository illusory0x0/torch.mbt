rm -rf target
cd libtorch_proxy
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=/your/path/to/libtorch ..
make -j
cp libtchproxy.so $HOME/.moon/lib
cd ../../
moon test --target native
