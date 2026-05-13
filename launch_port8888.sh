#!/bin/bash -l
#SBATCH -J Singularity_Jupyter_parallel_cuda
#SBATCH -N 1 # Nodes
#SBATCH -n 1 # Tasks
#SBATCH -c 1 # Cores assigned to each tasks
#SBATCH --time=0-01:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --qos=normal
#SBATCH --mail-user=<firstname>.<lastname>@uni.lu
#SBATCH --mail-type=BEGIN,END

#module load apptainer

export VENV="$HOME/.envs/sam2eszterdevel2" # ide hozza létre a venv-et (virtuális körny.)
export JUPYTER_CONFIG_DIR="$HOME/jupyter_sing/$SLURM_JOBID/"
export JUPYTER_PATH="$VENV/share/jupyter":"$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_sing/$SLURM_JOBID"
export XDG_RUNTIME_DIR="" # ennek itt kell lennie fenn

mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $IPYTHONDIR

export APPTAINER_BIND="/nas:/nas" # azon kívül, ahonnan indítod, mi lásson?

export JUPYTER_ALLOW_INSECURE_WRITES=1
apptainer instance start --nv /home/molnarester/Singularity_files/ubi22_decord_cmake_cu121.sif sam2eszter # indítja a példányt a megadott helyről. sam2eszter a példány neve


if [ ! -d "$VENV" ];then
    # For some reasons, there is an issue with venv -- using virtualenv instead
    echo "creating environment '$VENV'"
    apptainer exec --nv instance://sam2eszter python3.10 -m virtualenv -p python3.10 $VENV --system-site-packages
    apptainer run --nv instance://sam2eszter $VENV "python3 -m pip install --upgrade setuptools pip"
    #apptainer run --nv instance://sam2eszter $VENV "python3 -m pip install -e ."
    
    #apptainer run --nv instance://sam2eszter $VENV "python3 -m pip install nvidia-pyindex"
    #apptainer run --nv instance://sam2eszter $VENV "python3 -m pip install --upgrade nvidia-tensorrt"
    apptainer run --nv instance://sam2eszter $VENV "python3 -m pip install numpy scikit-image scikit-learn pandas matplotlib plotly"
    
    apptainer run --nv instance://sam2eszter $VENV "python3 -m pip install tensorflow==2.12.*"
    apptainer run --nv instance://sam2eszter $VENV "python3 -m pip install jupyter ipyparallel"
    #apptainer run --nv instance://sam2eszter $VENV "python3 -m pip install jupyter ipyparallel cgroup-utils"
    apptainer run --nv instance://sam2eszter $VENV "python3 -m ipykernel install --sys-prefix --name HPC_NRSFM_PYTORCH_ENV_IPYPARALLEL_CUDA --display-name HPC_NRSFM_PYTORCH_ENV_IPYPARALLEL_CUDA"

fi

#create a new ipython profile appended with the job id number
profile=job_${SLURM_JOB_ID}
apptainer run --nv instance://sam2eszter $VENV "ipython profile create --parallel ${profile}"

## Start Controller and Engines
#
apptainer run --nv instance://sam2eszter $VENV "ipcontroller --ip="*" --profile=${profile}" &
sleep 10

##srun: runs ipengine on each available core
apptainer run --nv instance://sam2eszter $VENV "ipengine --profile=${profile} --location=$(hostname)" &
sleep 25

export XDG_RUNTIME_DIR=""



JUPYTER_TOKEN="sammy" # jelszó

apptainer run --nv instance://sam2eszter $VENV "jupyter notebook --ip $(facter ipaddress) --no-browser --port 8888 --NotebookApp.token=${JUPYTER_TOKEN}" &

pid=$!
sleep 5s
#apptainer run --nv instance://sam2eszter $VENV "jupyter notebook list"
#apptainer run --nv instance://sam2eszter $VENV "jupyter --paths"
#apptainer run --nv instance://sam2eszter $VENV "jupyter kernelspec list"

wait $pid
echo "Stopping instance"
apptainer instance stop sam2eszter
