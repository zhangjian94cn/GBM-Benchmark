FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

COPY apt-install-and-clear.sh /usr/local/bin/apt-install-and-clear


# Per https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
# we need to add a new GPG key before running apt update.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Base scripts
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean
RUN apt-get update --fix-missing

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# core lib
COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

SHELL ["/bin/bash", "--login", "-c"]

# conda
COPY install/ubuntu_install_conda.sh /install/ubuntu_install_conda.sh
RUN bash /install/ubuntu_install_conda.sh
ENV PATH=/opt/conda/bin:${PATH}

# python package
COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh



# tvm
WORKDIR /workspace
RUN git clone --recursive https://github.com/apache/tvm tvm && \
    cd tvm && git checkout tags/v0.9.0 && pwd && mkdir build && \
    cp cmake/config.cmake build && \
    sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' build/config.cmake && \
    sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM \/usr\/lib\/llvm-11\/bin\/llvm-config)/g' build/config.cmake && \
    cd build && cmake .. && make -j$(nproc) && \
    cd .. && cd python && python setup.py install --user; 

# Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}

# Ensure the local libcuda have higher priority than the /usr/local/cuda/compact
# since the compact libcuda does not work on non-Tesla gpus
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}





