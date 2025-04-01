#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:30:15 2025

@author: gabriel.castejon
"""


import functions as fs
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# to run SPPARKS:
# in spyder:
# Open terminal on Applications (top left)
# go to PRIMME folder
# run the following command:
    # module load ufrc mkl/2023.2.0 gcc/12.2.0 openmpi/4.1.5 sypder/5.3.2
    # python test_run.py
    
fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=25.0, nsets=200, max_steps=10, future_steps=4, freq=(0.5,0.5))
# max_steps=100, offset_steps=1, future_steps=4, freq = (1,1), del_sim=False):
    
# fs.create_SPPARKS_dataset(size=[93, 93, 93], ngrains_rng=[256,256],kt=0.66,cutoff=0, nsets=200,max_steps=10, future_steps=4, freq=(0.5,1))
