#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:31:02 2024

@author: gabriel.castejon
"""

import functions as fs
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas
# from scipy.stats import multivariate_normal
# import scipy as sp
import shutil
import torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

w=2; h=2
sw=3; sh=2
if_leg = True

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    
def calculate_r2_and_ng_vs_time(data_file, model_type):
    """Calculates <R>^2 vs time and number of grains vs time.

    Args:
        data_file (str): Path to the h5py (PRIMME) or (SPPARKS) file.
        model_type (str): 'FULL' or 'SPLIT'

    Returns:
        tuple: (r2, t, ng)
            r2: average grain radius squared over time
            t: time points
            ng: number of grains over time
    """

    if model_type == 'FULL':
        with h5py.File(data_file, 'r') as f:
            # for k, key in enumerate(f.keys()):
            #     print("value: ", k, ", Key: ", key)
            # sim0 = f['sim0/']
            
            # euler_angles = f['sim0/euler_angles']
            # ims_id = f['sim0/ims_id']
            # miso_array = f['sim0/miso_array']
            # miso_matrix = f['sim0/miso_matrix']
            
            grain_areas = f['sim0/grain_areas'][:]
    elif model_type == 'SPLIT':
        grain_areas = np.load(data_file)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ng = (grain_areas != 0).sum(1)
    si = np.argmin(np.abs(ng - 410))
    grain_radii = np.cbrt(grain_areas * 3 / 4 / np.pi)
    grain_radii_avg = grain_radii.sum(1) / ng
    r2 = (grain_radii_avg ** 2)[:si] 
    t = np.arange(si)

    return r2, t, ng[:si] 

# Example Usage:
primme_file = '../3DPRIMME/PRIMME/data/primme_sz(93x93x93)_ng(16384)_nsteps(1000)_freq(1)_kt(0.66)_freq(0.5)_cut(0).h5'
spparks_file = '../3DPRIMME/PRIMME/data/trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(10)_kt(0.66)_freq(0.1)_cut(0).h5'


print("PRIMME Structure")

# with h5py.File(primme_file, 'r') as f:
#     miso_array = f['sim0/miso_array'][:]
#     miso_matrix = fs.miso_array_to_matrix(torch.from_numpy(miso_array[None,])).to(device)

with h5py.File(primme_file, 'r') as f:
    for k, key in enumerate(f.keys()):
        print("value: ", k, ", Key: ", key)
    print("Going inside sim0")
    sim0 = f['sim0/']
    for k, key in enumerate(sim0.keys()):
        print("\tvalue: ", k, ", Key: ", key)
        if isinstance(k, dict):
            for k2, key2 in enumerate(sim0[key].keys()):
                print("\t\tvalue: ", k2, ", Key: ", key2)
print("SPPARKS Structure")
# fs.generate_SPPARKS_miso_matrix(spparks_file)
with h5py.File(spparks_file, 'r') as f:
    for k, key in enumerate(f.keys()):
        print("value: ", k, ", Key: ", key)
        if isinstance(k, dict):
            for k2, key2 in enumerate(f[key].keys()):
                print("\tvalue: ", k2, ", Key: ", key2)

        
        
fs.compute_grain_stats(spparks_file, gps='spparks')
with h5py.File(spparks_file, 'r') as f:
    for k, key in enumerate(f.keys()):
        print("value: ", k, ", Key: ", key)

r2_primme, t_primme, ng_primme = calculate_r2_and_ng_vs_time(primme_file, 'FULL')
# r2_spparks, t_spparks, ng_spparks = calculate_r2_and_ng_vs_time(spparks_file, 'FULL')

# Plotting (using your existing plotting logic)
plt.figure(figsize=[sw,sh], dpi=600)


plt.rcParams['font.size'] = 8
plt.plot(t_primme, r2_primme)
# plt.plot(t_spparks, ng_spparks,'--')
plt.xlabel('Time (Unitless)')
plt.ylabel('$<R>^2$ (Pixels)')
plt.xlim([0,t_primme[-1]])
plt.ylim([0,100])
if if_leg: plt.legend(['PRIMME','SPPARKS'])
plt.savefig('./plots/3d_r2_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()



# ... (Rest of your plotting code)
    



# Code to refer for

"""
# File Paths to Use:

path_to_data = "../3DPRIMME/PRIMME/data/"
joseph_spparks = "../3DPRIMME/PRIMME/data/spparks_expected.npy"

spparks_to_test_1 = 

# 'primme_sz(93x93x93)_ng(16384)_nsteps(1000)_freq(1)_kt(0.66)_freq(0.1)_cut(0).h5'
# 'primme_sz(93x93x93)_ng(16384)_nsteps(1000)_freq(1)_kt(0.66)_freq(0.5)_cut(0).h5'
# 'primme_sz(93x93x93)_ng(16384)_nsteps(1000)_freq(1)_kt(0.66)_freq(1.0)_cut(0).h5'
# 'primme_sz(93x93x93)_ng(256)_nsteps(1000)_freq(1)_kt(0.66)_freq(1.0)_cut(0).h5'
# 'primme_sz(93x93x93)_ng(8192)_nsteps(1000)_freq(1)_kt(0.66)_freq(0.5)_cut(0).h5'
# 'trainset_spparks_sz(128x128x128)_ng(512-512)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5'
# 'trainset_spparks_sz(128x128x128)_ng(512-512)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(1.0)_cut(0).h5'
# 'trainset_spparks_sz(64x64x64)_ng(1024-1024)_nsets(200)_future(4)_max(10)_kt(0.66)_freq(0.1)_cut(0).h5'
# 'trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(10)_kt(0.66)_freq(0.1)_cut(0).h5'
# 'trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(10)_kt(0.66)_freq(0.5)_cut(0).h5'
# 'trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(10)_kt(0.66)_freq(1.0)_cut(0).h5'
# 'trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5'


### 3D isotropic <R>^2vs time, number of grains vs time (MF, SPPARKS) #!!!

with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
    grain_areas = f['sim0/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
si = np.argmin(np.abs(ng-410))
grain_radii = np.cbrt(grain_areas*3/4/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2_mf = (grain_radii_avg**2)[:si] #square after the mean
t_mf = np.arange(si)
p_mf = np.polyfit(t_mf, r2_mf, deg=1)[0]
ng_mf = ng[:si]

grain_areas = np.load('./data/spparks_grain_areas_128p3_8192.npy')
ng = (grain_areas!=0).sum(1)
si = np.argmin(np.abs(ng-410))
grain_radii = np.cbrt(grain_areas*3/4/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2_mcp = (grain_radii_avg**2) #square after the mean
p = np.polyfit(np.arange(si), r2_mcp[:si], deg=1)[0]
scale = p/p_mf
t_mcp = np.arange(len(r2_mcp))*scale
ng_mcp = ng

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(t_mf, r2_mf)
plt.plot(t_mcp, r2_mcp,'--')
plt.xlabel('Time (Unitless)')
plt.ylabel('$<R>^2$ (Pixels)')
plt.xlim([0,t_mf[-1]])
plt.ylim([0,100])
if if_leg: plt.legend(['MF','MCP'])
plt.savefig('./plots/3d_r2_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()



### 3D isotropic average number of sides through time (MF, SPPARKS) #!!!

with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
    gsa_mf = f['sim0/grain_sides_avg'][:]
    
gsa_mcp = np.load('./data/spparks_grain_sides_avg_128p3_8192.npy')

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(t_mf, gsa_mf[:len(t_mf)])
plt.plot(t_mcp, gsa_mcp,'--')
plt.xlim([0, t_mf[-1]])
plt.xlabel('Time (Unitless)')
plt.ylabel('Avg Number \nof Sides')
if if_leg: plt.legend(['MF','MCP'])
plt.savefig('./plots/3d_num_sides_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()



### 3D isotropic normalized radius distribution (MF, SPPARKS) #!!!
pf_r2_dist = scipy.io.loadmat('./data/pf/3DPFDataRadDist.mat')
x_pf = pf_r2_dist['xr1'][0]
h_pf = pf_r2_dist['RadDist'][0]

for num_grains in [2000, 1500, 1000]:
    
    with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
    n = (grain_areas!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    a = grain_areas[i]
    r = np.cbrt(a[a!=0]*3/4/np.pi)
    rn = r/np.mean(r)
    h_mf, x_edges = np.histogram(rn, bins='auto', density=True)
    x_mf = x_edges[:-1]+np.diff(x_edges)/2
    n_mf = len(rn)
    
    grain_areas = np.load('./data/spparks_grain_areas_128p3_8192.npy')
    n = (grain_areas!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    a = grain_areas[i]
    r = np.cbrt(a[a!=0]*3/4/np.pi)
    rn = r/np.mean(r)
    h_mcp, x_edges = np.histogram(rn, bins='auto', density=True)
    x_mcp = x_edges[:-1]+np.diff(x_edges)/2
    n_mcp = len(rn)
    
    plt.figure(figsize=[sw,sh], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_mf, h_mf, '-')
    plt.plot(x_mcp, h_mcp, '--')
    plt.plot(x_pf, h_pf, '.')
    plt.xlabel('$R/<R>$ - Normalized Radius')
    plt.ylabel('Frequency')
    if if_leg: plt.legend(['MF, $N_G$ - %d'%n_mf, 'MCP, $N_G$ - %d'%n_mcp, 'Yadav 2018 (2)'], fontsize=7)
    plt.savefig('./plots/3d_r_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
    plt.show()




## 3D isotropic number of sides distribution (MF, SPPARKS) #!!!
pf_numsides_dist = scipy.io.loadmat('./data/pf/3DPFDataNumSides.mat')
x_pf = pf_numsides_dist['xf1'][0, :35]
h_pf = pf_numsides_dist['NumSides'][0, :35]

for num_grains in [2000, 1500, 1000]:

    with h5py.File('./data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(4)_numnei(64)_cut(0).h5', 'r') as f:
        grain_sides = f['sim0/grain_sides'][:]
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,35)-0.5
    h_mf, x_edges = np.histogram(s, bins=bins, density=True)
    x_mf = x_edges[:-1]+np.diff(x_edges)/2
    n_mf = len(s)
    
    grain_sides = np.load('./data/spparks_grain_sides_128p3_8192.npy')
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,35)-0.5
    h_mcp, x_edges = np.histogram(s, bins=bins, density=True)
    x_mcp = x_edges[:-1]+np.diff(x_edges)/2
    n_mcp = len(s)
    
    plt.figure(figsize=[sw,sh], dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_mf, h_mf, '-')
    plt.plot(x_mcp, h_mcp, '--')
    plt.plot(x_pf, h_pf, '.')
    plt.xlabel('Number of Sides')
    plt.ylabel('Frequency')
    if if_leg: plt.legend(['MF, $N_G$ - %d'%n_mf, 'MCP, $N_G$ - %d'%n_mcp, 'Yadav 2018 (2)'], fontsize=7)
    plt.savefig('./plots/3d_num_sides_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
    plt.show()

"""