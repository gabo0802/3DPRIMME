#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:26:49 2023

@author: joseph.melville
"""



import PRIMME as fsp
import functions as fs
import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = './data/trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5'

num_eps = 1000
test_epocs = [5, 10, 20, 50, 100, 200, 300, 500]
# modelname = fsp.train_primme(trainset, num_eps=num_eps, obs_dim=9, act_dim=9, lr=5e-5, reg=1, if_miso=False, plot_freq=1, multi_epoch_safe=False)

modelname = './data/model_dim(3)_sz(9_9)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_freq(0.5)_cut(0).h5'

# was 93^3
ic, ea, _ = fs.voronoi2image(size=[96,96,96], ngrain=2**10)
ma = fs.find_misorientation(ea, mem_max=1) 

np.save("./data/ic.npy", ic), np.save("./data/ea.npy", ea), np.save("./data/ma.npy", ma)
#ic, ea, ma = np.load("./data/ic.npy"), np.load("./data/ea.npy"), np.load("./data/ma.npy")


# for epoch in range(100, num_eps -1, 100):
#     cur_model = f"{modelname[:-3]}_at_epoch({epoch}).h5"
#     fp = fsp.run_primme(ic, ea, nsteps=200, modelname=cur_model, miso_array=ma, pad_mode='circular', if_miso=False, plot_freq=1)
#     fs.compute_grain_stats(fp)
#     fs.make_videos(fp, multi_res=True, epoch=epoch)
#     fs.make_time_plots(fp, multi_res=True, epoch=epoch)

for epoch in test_epocs:
    modelname = fsp.train_primme(trainset, num_eps=epoch, obs_dim=9, act_dim=9, lr=5e-5, reg=1, if_miso=False, plot_freq=1, multi_epoch_safe=False)
    fp = fsp.run_primme(ic, ea, nsteps=200, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=False, plot_freq=1)
    fs.compute_grain_stats(fp)
    fs.make_videos(fp, multi_res=True, epoch=epoch)
    fs.make_time_plots(fp, multi_res=True, epoch=epoch)
fp= "./data/primme_sz(93x93x93)_ng(16384)_nsteps(1000)_freq(1)_kt(0.66)_freq(0.1)_cut(0).h5"

#Last one
# fp = fsp.run_primme(ic, ea, nsteps=200, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=False, plot_freq=1)
# fs.compute_grain_stats(fp)
# fs.make_videos(fp)
# fs.make_time_plots(fp)

