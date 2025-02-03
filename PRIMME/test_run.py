#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:26:49 2023

@author: joseph.melville
"""



import PRIMME as fsp
import functions as fs
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# import pandas
# from scipy.stats import multivariate_normal
# import scipy as sp
# import shutil
import torch
# import scipy.io
# from tqdm import tqdm
from matplotlib.patches import Rectangle
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# to run SPPARKS:
# in spyder:
# Open terminal on Applications (top left)
# go to PRIMME folder
# run the following command:
    # module load ufrc mkl/2023.2.0 gcc/12.2.0 openmpi/4.1.5 sypder/5.3.2
    # python test_run.py
    
# def create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=25.0, nsets=200, 
# max_steps=100, offset_steps=1, future_steps=4, freq = (1,1), del_sim=False):

# trainset = fs.create_SPPARKS_dataset(size=[93, 93, 93], ngrains_rng=[256,256],kt=0.66,cutoff=0, nsets=200,max_steps=10, future_steps=4, freq=(0.5,1))

trainset = './data/trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5'

# I have 3 trainsets like this, only difference is the freq attribute, will be testing a primme model with each to determine diff:
# trainset = './data/trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(10)_kt(0.66)_freq(0.5)_cut(0).h5'

#trainset = './data/trainset_spparks_sz(64x64x64)_ng(1024-1024)_nsets(200)_future(4)_max(10)_kt(0.66)_freq(0.1)_cut(0).h5'
# trainset = './data/trainset_spparks_sz(128x128x128)_ng(512-512)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5'

num_eps = 2000
modelname = fsp.train_primme(trainset, num_eps=num_eps, obs_dim=9, act_dim=9, lr=5e-5, reg=1, if_miso=False, plot_freq=1)
#modelname = './data/model_dim(3)_sz(13_13)_lr(5e-05)_reg(1)_ep(3000)_kt(0.66)_freq(0.5)_cut(0).h5'

# was 93^3
ic, ea, _ = fs.voronoi2image(size=[96,96,96], ngrain=2**14)
ma = fs.find_misorientation(ea, mem_max=1) 

np.save("./data/ic.npy", ic), np.save("./data/ea.npy", ea), np.save("./data/ma.npy", ma)
#ic, ea, ma = np.load("./data/ic.npy"), np.load("./data/ea.npy"), np.load("./data/ma.npy")


for epoch in range(0, num_eps -1, 100):
    cur_model = f"{modelname[:-3]}_at_epoch({epoch}).h5"
    fp = fsp.run_primme(ic, ea, nsteps=200, modelname=cur_model, miso_array=ma, pad_mode='circular', if_miso=False, plot_freq=1)
    fs.compute_grain_stats(fp)
    fs.make_videos(fp, multi_res=True, epoch=epoch)
    fs.make_time_plots(fp, multi_res=True, epoch=epoch)

#Last one
fp = fsp.run_primme(ic, ea, nsteps=200, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=False, plot_freq=1)
#fp= "./data/primme_sz(93x93x93)_ng(16384)_nsteps(1000)_freq(1)_kt(0.66)_freq(0.1)_cut(0).h5"

fs.compute_grain_stats(fp)
fs.make_videos(fp)
fs.make_time_plots(fp)

