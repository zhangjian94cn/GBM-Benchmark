
# zhangjian
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

conda activate benchmark && \
pip install matplotlib xgboost sklearn pandas


_intelexHome=/home/scikit-learn-intelex

if [ -d $_intelexHome ]; then
    echo "scikit learn intelex found"
else
	cd /home
    git clone https://github.com/intel/scikit-learn-intelex.git
fi

# export DALROOT=/home/oneDAL/__release_lnx/daal/latest
# export DALROOT=/home/oneDAL/__release_lnx_gnu/daal/latest
# export DALROOT=$CONDA_PREFIX
export DALROOT=/opt/intel/oneapi/dal/latest/

export MPIROOT=/opt/intel/oneapi/mpi/latest/
# export MPIROOT=$CONDA_PREFIX

export OFF_ONEDAL_IFACE=1

cd $_intelexHome;
python setup.py develop --no-deps
# python setup.py install --single-version-externally-managed --record=record.txt