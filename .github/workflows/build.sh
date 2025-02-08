rm -rf target
cd libtorch_proxy
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
make -j
cp libtchproxy.so ../../
cd ../../