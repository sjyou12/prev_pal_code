#/bin/sh
#
# For PAL-XFEL scattering exp analysis 
#
# $1      -- name of partition
# $2      -- # of core
# $3      -- name of input file
# 
# Now check for the input file
#

if [ "x$3" = "x" ]
then
   echo "Usage: $0 (partition_name) (# of Core) input-file[.inp] "
   exit 1
fi

# Check if the input file exists with extension .inp

if ! test -e $3.py
then
	echo "$3.py DOES NOT exist"
	echo "This job is not submitted"
	exit 1
fi

# Start creating the input file
cat <<END >>$3.job
#!/bin/sh
# SBATCH --nodelist=node02
#SBATCH -N 1
#SBATCH -p "$1"
#SBATCH --error="$3.err.%j"
#SBATCH --output="$3.log.%j"
#SBATCH -J 1d_generator.$3
#SBATCH -n $2
#SBATCH -A $USER

cd \$SLURM_SUBMIT_DIR

echo Running on host \`hostname\`
echo Time is \`date\`
echo Directory is \`pwd\`
echo Job Name is $3
# echo This jobs runs on the following processors:
# echo \`cat \$SLURM_JOB_NODELIST\`

# NPROCS=\`wc -l < \$SLURM_JOB_NODELIST\`

source /home/anaconda3/etc/profile.d/conda.sh
conda activate anal_data
export PYTHONPATH=/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/:$PYTHONPATH
python3.7 $3.py 

exit 0
END


echo "This job will request $2 processor(s)"
  
# Now submit it.
sbatch $3.job
rm -f $3.job

# Wait a short time and exit
sleep 1

