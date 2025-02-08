cp .github/workflows/moon.pkg.asan.json torch/moon.pkg.json
rm -rf target
cd libtorch_proxy
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch -DUSE_ASAN=ON ..
make -j
cp libtchproxy.so ../../
cd ../../