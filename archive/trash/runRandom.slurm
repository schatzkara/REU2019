#!/bin/bash

## Define resources to allocate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
## increase tasks and gpu to 2 for 2 gpus, increase nodes if using more than 2 gpus (2 gpus per node)
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -w evc11

## Nothing should take longer than 5 days
#SBATCH --time=120:00:00

## Put name of job here (shows up in queue)
#SBATCH --job-name=RandomCrossView

## Fill in <NAME> here (name of files created for logs)
#SBATCH --error=cifar100.%J.err
#SBATCH --output=cifar100.%J.out

## Select File to run
export file="RunRandom.py"
## Define any program arguments
export args=""
## Folder name, should be same as job-name
export job_name="RandomCrossView"

## Select how logs get stored.
export log_dir="/home/crcvreu.student4/job_logs/$job_name-$SLURM_JOB_ID"
mkdir $log_dir
export debug_logs="$log_dir/job_$SLURM_JOB_ID.log"
export benchmark_logs="$log_dir/job_$SLURM_JOB_ID.log"

## Load Modules
module load anaconda/anaconda3
module load cuda/cuda-9.0
module load openmpi/openmpi-4.0.0-ic-2019.1.144

## Activate anaconda env
source activate reu-gpu

## Enter Working Directory ##
cd $SLURM_SUBMIT_DIR

## Create Log File ##
echo "Python version: $(python --version)" >> $debug_logs
echo "Slurm working directory: $SLURM_SUBMIT_DIR" >> $debug_logs
echo "JobID: $SLURM_JOB_ID" >> $debug_logs
echo "Running on $SLURM_NODELIST" >> $debug_logs
echo "Running on $SLURM_NNODES nodes." >> $debug_logs
echo "Running on $SLURM_NPROCS processors." >> $debug_logs
echo "Current working directory is `pwd`" >> $debug_logs

## Module debugging ##
echo "Modules loaded: $(module list)" >> $debug_logs
echo "mpirun location: $(which mpirun)" >> $debug_logs
echo "python location: $(which python)" >> $debug_logs

echo "Starting time: $(date)" >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Run job ##
time mpirun -np $SLURM_NTASKS nvidia-smi && python $file $args
sleep 3

echo "Ending time: $(date)" >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv $job_name.$SLURM_JOB_ID.err $log_dir
mv $job_name.$SLURM_JOB_ID.out $log_dir
