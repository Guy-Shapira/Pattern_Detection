#!/bin/bash


NUM_NODES=1
NUM_CORES=1
NUM_GPUS=1
JOB_NAME="noise_runs"
MAIL_USER="guy.shapira@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/anaconda3
CONDA_ENV=inv
sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
	--begin=now+10\
<<EOF
#!/bin/bash
export PYTHONPATH=. 
echo $PYTHONPATH
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"
# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV
# Run python with the args to the script

python Model/model_OpenCEP.py --data_path=StarPilot/GamesExp/ --events_path=StarPilot/EventsExp --pattern_path=Patterns/pattern28_50_ratings.csv --early_knowledge=True --wandb_name=full_sigma_seed0 --run_mode=full --noise_flag=True --mu=0 --sigma=1
python Model/model_OpenCEP.py --data_path=StarPilot/GamesExp/ --events_path=StarPilot/EventsExp --pattern_path=Patterns/pattern28_50_ratings.csv --early_knowledge=True --wandb_name=full_sigma5_seed0 --run_mode=full --noise_flag=True --mu=0 --sigma=5
python Model/model_OpenCEP.py --data_path=StarPilot/GamesExp/ --events_path=StarPilot/EventsExp --pattern_path=Patterns/pattern28_50_ratings.csv --early_knowledge=True --wandb_name=full_sigma10_seed0 --run_mode=full --noise_flag=True --mu=0 --sigma=10
python Model/model_OpenCEP.py --data_path=StarPilot/GamesExp/ --events_path=StarPilot/EventsExp --pattern_path=Patterns/pattern28_50_ratings.csv --early_knowledge=True --wandb_name=full_simga20_seed0 --run_mode=full --noise_flag=True --mu=0 --sigma=20
echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF