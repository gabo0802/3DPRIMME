#!/bin/bash

# Run the Python script
python3 /lustre/blue/joel.harley/gabriel.castejon/3DPRIMME/PRIMME/run_primme.py
# Add arguments to the script
--trainset ./data/trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5 \
--modelname \
--num_eps 2000 \
--obs_dim 9 \
--act_dim 9 \
--lr 5e-5 \
--reg 1 \
--if_miso \
--plot_freq 1 \
--voroni_loaded \
--ic ./data/ic.npy \
--ea ./data/ea.npy \
--ma ./data/ma.npy \
--size 93 \
--dimension 3 \
--ngrain 16384 \
--primme ./data/primme_sz(93x93x93)_ng(16384)_nsteps(1000)_freq(1)_kt(0.66)_freq(0.1)_cut(0).h5 \
--multi_run False \
--nsteps 200 \
--pad_mode circular