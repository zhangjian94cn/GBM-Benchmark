
# 

export DALROOT=/workspace/oneDAL/__release_lnx_gnu/daal/latest
export MPIROOT=/opt/intel/oneapi/mpi/latest
export OFF_ONEDAL_IFACE=1

sklearnHome=/workspace/scikit-learn-intelex

cd $sklearnHome;
python setup.py develop --no-deps