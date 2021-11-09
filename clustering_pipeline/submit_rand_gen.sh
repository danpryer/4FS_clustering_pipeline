#!/bin/bash -l
#SBATCH  -J FourFS_rand_gen     # job name
#SBATCH --ntasks=16            # use this many tasks
#SBATCH -o /cosma/home/dp004/dc-prye1/4fs_clustering_project/cosma_output/out_file.%J.out
#SBATCH -e /cosma/home/dp004/dc-prye1/4fs_clustering_project/cosma_output/err_file.%J.err
#SBATCH -p cosma7 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004    # project id
#SBATCH --exclusive
#SBATCH -t 8:00:00 # max run time
#SBATCH --mail-type=END # notifications for job done & fail
#SBATCH --mail-user=d.pryer@sussex.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#load the modules used to build your program.
module load python/3.6.5
module load gnu_comp/7.3.0
module load openmpi/3.0.1

##rand_multi=50

# Run the program
mpiexec -n $SLURM_NTASKS python3 /cosma/home/dp004/dc-prye1/4fs_clustering_project/clustering_pipeline/parallel_rand_gen.py ## $rand_multi
