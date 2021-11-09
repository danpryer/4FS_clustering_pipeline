#!/bin/bash -l
#SBATCH  -J FourFS_pk_zbins     # job name
#SBATCH --ntasks=1            # use this many tasks
#### SBATCH --ntasks-per-node=8   # use this many tasks on a node
#### SBATCH --cpus-per-task=28    # use this many threads per task
#SBATCH -o /cosma/home/dp004/dc-prye1/4fs_clustering_project/cosma_output/out_file.%J.out
#SBATCH -e /cosma/home/dp004/dc-prye1/4fs_clustering_project/cosma_output/err_file.%J.err
#SBATCH -p cosma7 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004    # project id
#SBATCH --exclusive
#SBATCH -t 20:00:00 # max run time
#SBATCH --mail-type=END # notifications for job done & fail
#SBATCH --mail-user=d.pryer@sussex.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#load the modules used to build your program.
module load python/3.6.5
module load gnu_comp/7.3.0
module load openmpi/3.0.1


tracer='QSO'  # ['BG', 'LRG', 'QSO', 'LyA']
catalogue='output' # ['input', 'output']
Nmesh=1024   # FFT mesh resolution, [128, 256, 512, 1024...]
Nbins_Z=6  # number of redshift bins. Recommended for [BG, LRG, QSO, LyA] are [6, 11, 12, 10]

window='tsc' # ['cic', 'tsc']
interlaced=1  # 0=False, 1=True
compensated=1  # 0=False, 1=True

real_or_redshift='redshift' # working in real or redshift space coords ['real', 'redshift']

dk=0.05 # linear bin width. Recommended for [BG, LRG, QSO, LyA] are [0.05, 0.04, 0.025, 0.025]



# Run the program
mpiexec -n $SLURM_NTASKS python3 /cosma/home/dp004/dc-prye1/4fs_clustering_project/clustering_pipeline/powerspec_zbins.py $tracer $catalogue $Nmesh $Nbins_Z $window $interlaced $compensated $real_or_redshift $dk
