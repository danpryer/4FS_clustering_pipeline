#!/bin/bash -l
#SBATCH --array=[0-7]           # run as array job
#SBATCH  -J FourFS_powerspec     # job name
#SBATCH --ntasks=1            # use this many tasks
#### SBATCH --ntasks-per-node=8   # use this many tasks on a node
#### SBATCH --cpus-per-task=28    # use this many threads per task
#SBATCH -o /cosma/home/dp004/dc-prye1/4fs_clustering_project/cosma_output/out_file.%J.out
#SBATCH -e /cosma/home/dp004/dc-prye1/4fs_clustering_project/cosma_output/err_file.%J.err
#SBATCH -p cosma7 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004    # project id
#SBATCH --exclusive
#SBATCH -t 10:00:00 # max run time
#SBATCH --mail-type=END # notifications for job done & fail
#SBATCH --mail-user=d.pryer@sussex.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#load the modules used to build your program.
module load python/3.6.5
module load gnu_comp/7.3.0
module load openmpi/3.0.1


# Run the program
mpiexec -n $SLURM_NTASKS python3 /cosma/home/dp004/dc-prye1/4fs_clustering_project/clustering_pipeline/powerspec.py $SLURM_ARRAY_TASK_ID
# python3 /cosma/home/dp004/dc-prye1/4fs_clustering_project/clustering_pipeline/powerspec.py $SLURM_ARRAY_TASK_ID
