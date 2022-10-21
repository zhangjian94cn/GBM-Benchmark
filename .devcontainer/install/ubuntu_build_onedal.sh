#!/bin/bash

## Copyright 2022 Intel Corporation
##
##  Content:
##     scripts for oneAPI Data Analytics Library building
##
##                                           by kunpeng: kunpeng.jiang@intel.com
##                                           modified by: zhangjian
##******************************************************************************


#Please set your oneDal source home and intel oneapi toolkit home properly



cd /home
git clone https://github.com/oneapi-src/oneDAL.git

_oneDalHome=/home/oneDAL
_intelOneapiHome=/opt/intel/oneapi

#Suppose you're using anaconda and use env named oneDal for oneDal building
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda create -n benchmark python=3.8 -y && \
conda activate benchmark && \
conda install -y impi-devel cython jinja2 numpy clang-tools pybind11 -c intel -c conda-forge && \
conda install -y openjdk=11.0.13

#Find dpcpp, which is necessary for target onapi
source $_intelOneapiHome/compiler/latest/env/vars.sh
if [ ! command -v dpcpp &> /dev/null ]; then
	echo "dpcpp is not installed properly, please check the install path"
	exit
fi

#Setup mklgpufpk
_mklgpuPATH=$_oneDalHome/__deps/mklgpufpk
export CPATH=$_mklgpuPATH/lnx/include:$CPATH

##Find or install mklgpufpk
if [ -d $_mklgpuPATH ]; then
	echo "mklgpu found!"
else
	$_oneDalHome/dev/download_micromkl.sh
fi

#Setup tbb
_tbbPATH=$_oneDalHome/__deps/tbb
#export CPATH=$_tbbPATH/lnx/include:$CPATH

##Find or install tbbfpk
if [ -d $_tbbPATH ]; then
	echo "tbb found!"
else
	$_oneDalHome/dev/download_tbb.sh
fi

#Setup openjdk path
#Which is necessary for daal
export CPATH=/opt/conda/envs/benchmark/include/linux:/opt/conda/envs/benchmark/include:$CPATH

#build from source
#Attention!! You have to build daal before build oneapi
cd $_oneDalHome;
# make daal -j1  PLAT=lnx32e COMPILER=gnu && \
# 	make oneapi -j1 PLAT=lnx32e COMPILER=gnu

make daal -j$(expr $(nproc) - 2)  PLAT=lnx32e


# cp $_intelOneapiHome/dal/latest/lib/intel64/* /opt/conda/envs/benchmark/lib/
# cp $_intelOneapiHome/tbb/latest/lib/intel64/* /opt/conda/envs/benchmark/lib/

cp $_oneDalHome/__release_lnx/daal/latest/lib/intel64/* /opt/conda/envs/benchmark/lib/
cp $_oneDalHome/__release_lnx/tbb/latest/lib/intel64/* /opt/conda/envs/benchmark/lib/

