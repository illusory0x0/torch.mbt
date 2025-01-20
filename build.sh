rm -rf target
cd libtorch_proxy
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=/your/path/to/libtorch ..
make -j
cp libtchproxy.so ../../
cd ../../
LD_LIBRARY_PATH=. moon test --target native