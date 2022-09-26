#!/bin/bash

## Copyright 2022 Intel Corporation
##
##  Content:
##     scripts for oneAPI Data Analytics Library building
##
##                                           by kunpeng: kunpeng.jiang@intel.com
##******************************************************************************


#Please set your oneDal source home and intel oneapi toolkit home properly

_oneDalHome=/workspace/oneDAL
_intelOneapiHome=/opt/intel/oneapi

#Suppose you're using anaconda and use env named oneDal for oneDal building
__conda_setup="$('/opt/conda/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
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
conda activate oneDal

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
export CPATH=/opt/conda/envs/oneDal/include/linux:/opt/conda/envs/oneDal/include:$CPATH

#build from source
#Attention!! You have to build daal before build oneapi
cd $_oneDalHome;
make daal -j$(expr $(nproc) - 2)  PLAT=lnx32e COMPILER=gnu && \
	make oneapi -j$(expr $(nproc) - 2) PLAT=lnx32e COMPILER=gnu