
# zhangjian

cd /home
git clone https://github.com/intel/scikit-learn-intelex.git

# export DALROOT=/home/oneDAL/__release_lnx/daal/latest
# export DALROOT=/home/oneDAL/__release_lnx_gnu/daal/latest
# export DALROOT=$CONDA_PREFIX
export DALROOT=/opt/intel/oneapi/dal/latest/

export MPIROOT=/opt/intel/oneapi/mpi/latest/
# export MPIROOT=$CONDA_PREFIX

export OFF_ONEDAL_IFACE=1

sklearnHome=/home/scikit-learn-intelex

cd $sklearnHome;
python setup.py develop --no-deps
# python setup.py install --single-version-externally-managed --record=record.txt