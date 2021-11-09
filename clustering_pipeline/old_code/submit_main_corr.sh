#!/bin/bash -l
#SBATCH  -J FourFS_2pcf     # job name
#SBATCH --ntasks=1            # use this many task
#SBATCH --cpus-per-task=28    # use this many threads per task
#SBATCH -o /cosma/home/dp004/dc-prye1/4fs_clustering_project/cosma_output/out_file.%J.out
#SBATCH -e /cosma/home/dp004/dc-prye1/4fs_clustering_project/cosma_output/err_file.%J.err
#SBATCH -p cosma7 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004    # project id
#SBATCH --exclusive
#SBATCH -t 72:00:00 # max run time
#SBATCH --mail-type=END # notifications for job done & fail
#SBATCH --mail-user=d.pryer@sussex.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#load the modules used to build your program.
module load python/3.6.5


# Run the program
python3 /cosma/home/dp004/dc-prye1/4fs_clustering_project/clustering_pipeline/main_corr.py
