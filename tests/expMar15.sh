#!/bin/bash
SUBDIR="Mar15_6.23_9d3/"
N_EPISODES=5000

# python script_v3.99.py -cp 10 -cn 10 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v4.0.py -cp 0.1 -cn 1. --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# python script_v6.11.py -cp 0.1 -cn 1. --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# python script_v6.20.py -cp 0.05 -cn 0.3 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
python script_v6.23_0d.py -cp 0.05 -cn 5. --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# python script_v6.21.py -cp 0. -cn 0. --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v6.1.py -cp 1. -cn 0.1 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v6.1.py -cp 0.1 -cn 1. --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v4.0.py -cp 1 -cn 0.1 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v4.0.py -cp 0.1 -cn 0.1 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 10


# python script_v4.0.py -cp 0.1 -cn 1. --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# # python script_v4.0d.py -cp 0.1 -cn 1 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v4.0.py -cp 1 -cn 0.1 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v4.0.py -cp 0.1 -cn 0.1 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 10

# python script_v3.98.py -cp 10 -cn 10 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v3.98.py -cp 10 -cn 50 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v3.98.py -cp 50 -cn 10 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 10

# python script_v3.98.py -cp 10 -cn 10 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v3.98.py -cp 10 -cn 50 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 3
# python script_v3.98.py -cp 50 -cn 10 --n_ep=$N_EPISODES --data_subdir=$SUBDIR &
# sleep 10