cd /home
git clone --recursive https://github.com/apache/tvm tvm && \
cd tvm && git checkout tags/v0.9.0 && pwd && mkdir build && \
cp cmake/config.cmake build && \
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM \/usr\/lib\/llvm-11\/bin\/llvm-config)/g' build/config.cmake

cd build && cmake .. && make -j$(nproc) && \
cd .. && cd python && python setup.py install --user; 