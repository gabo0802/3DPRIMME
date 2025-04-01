#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""

# IMPORT LIBRARIES
import numpy as np
import functions as fs
import torch
import h5py
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class TrainingDataset(Dataset):
    def __init__(self,data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class PRIMME(nn.Module):
    def __init__(self, obs_dim=17, act_dim=17, energy_dim=3, pad_mode='circular', learning_rate=5e-5, reg=1, num_dims=2, if_miso=False, if_training=True):
        super(PRIMME, self).__init__()
    
        # self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.energy_dim = energy_dim
        self.pad_mode = pad_mode
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_dims = num_dims
        self.if_miso = if_miso
        self.training_loss = []
        self.validation_loss = []
        self.training_acc = []
        self.validation_acc = []
        self.if_training = if_training
        
        self.logx = []
        self.logy = []
        
        # DEFINE NEURAL NETWORK (originally 21*21*4, 21*21*2, 21*21, self.act_dim ** self.num_dums)
        norm = 128
        self.f1 = nn.Linear(self.obs_dim ** self.num_dims, norm) # Replace with: 512, 256, 128, 64, all of 21*21*4
        self.f2 = nn.Linear(norm, norm)
        self.f3 = nn.Linear(norm, norm)
        self.f4 = nn.Linear(norm, self.act_dim ** self.num_dims)
        self.dropout = nn.Dropout(p = 0.25) 
        self.BatchNorm1 = nn.BatchNorm1d(norm)
        self.BatchNorm2 = nn.BatchNorm1d(norm)
        self.BatchNorm3 = nn.BatchNorm1d(norm)

        # 21* 21 -> 512

        # DEFINE NEURAL NETWORK OPTIMIZATION
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=1e-5)
        self.loss_func = torch.nn.MSELoss()  # Mean squared error loss
        self.optimizer.zero_grad()  # Make all the gradients zero
                
    
    
    def forward(self, x):
        # def forward: Run input X through the neural network
        #   Inputs--
        #        x: microstructure features around a center pixel
        #   Outputs--
        #       y: "action likelihood" for the center pixel to flip to each the grain associated with each other pixel
        
        out = F.relu(self.f1(x))
        out = self.dropout(out)   
        out = self.BatchNorm1(out)
        out = F.relu(self.f2(out))
        out = self.dropout(out)
        out = self.BatchNorm2(out)
        out = F.relu(self.f3(out))
        out = self.dropout(out)
        out = self.BatchNorm3(out)
        # y  = torch.relu(self.f4(out))
        y  = torch.sigmoid(self.f4(out))
        
        return y
    
    
    def sample_data(self, h5_path='spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5'):
        #Extracts training and validation image sequences ("im_seq") and misorientation matricies ("miso_matrix") from "h5_path"
        #One image sequence extracted at a time, at random, given and 80/20, train/validation split
        #Calculates the batch size ("batch_sz") and number of iterations needed to iterate through a generator with that batch size ("num_iter")
        
        with h5py.File(h5_path, 'r') as f:
            i_max = f['ims_id'].shape[0]
            i_split = int(i_max*0.8)
            
            i_train = np.sort(np.random.randint(low=0, high=i_split, size=(1,)))
            batch = f['ims_id'][i_train,]
            miso_array = f['miso_array'][i_train,] 
            
            i_val = np.sort(np.random.randint(low=i_split, high=i_max, size=(1,)))
            batch_val = f['ims_id'][i_val,]
            miso_array_val = f['miso_array'][i_val,] 
         
        #Convert image sequences to Tenors and copy to "device"
        self.im_seq = torch.from_numpy(batch[0,].astype(float)).to(device)
        self.im_seq_val = torch.from_numpy(batch_val[0,].astype(float)).to(device)
        
        #Arrange misorientation matricies if indicated
        if self.if_miso:
            miso_array = miso_array[:, miso_array[0,]!=0] #cut out zeros, each starts with different number of grains
            miso_array = torch.from_numpy(miso_array.astype(float)).to(device)
            self.miso_matrix = fs.miso_array_to_matrix(miso_array)
            
            miso_array_val = miso_array_val[:, miso_array_val[0,]!=0] #cut out zeros, each starts with different number of grains
            miso_array = torch.from_numpy(miso_array_val.astype(float)).to(device)
            self.miso_matrix_val = fs.miso_array_to_matrix(miso_array)
        else:
            self.miso_matrix = None
            self.miso_matrix_val = None
    

        
    
    def train_step(self, im_seq, miso_matrix=None, unfold_mem_lim=4e9):
        #"im_seq" can be of shape=[1,1,dim0,dim1,dim2] or a sequence of shape=[num_future, 1, dim0, dim1, dim2]
        #Find the image after "im" given the trained model
        #Also calculates loss and accurate if given an image sequence
        #Calculates misorientation features of "miso_matrix" is given
        
        #Calculate input batch size to maintain memory usage limit 
        batch_sz = 5000 # 5000 10000 20000 50000 100000 200000 500000
        ''''
        Previous batching calculation: 
        int(unfold_mem_lim/((num_future)*self.act_dim**self.num_dims*self.energy_dim**self.num_dims*64)) #set to highest memory functions - "compute_energy_labels_gen"
        '''
        im = im_seq[0:1,]

        # Initialize variables and
        features = fs.compute_features(im, self.obs_dim, self.pad_mode)
        labels = fs.compute_labels(im_seq, self.act_dim, self.reg, self.pad_mode)
        labels = (labels - torch.min(labels)) / (torch.max(labels) - torch.min(labels))

        dataset = TrainingDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
        

        for features_batch, labels_batch in dataloader:
            # Only use neighborhoods that have more than one ID (have a potential to change ID)
            # features_batch has shape [B, 9, 9, 9]
            center = self.obs_dim // 2  # 9//2 = 4
            # Extract the center value for each patch; shape [B]
            current_ids = features_batch[:, center, center, center]  
            
            # Create a mask: compare every voxel in each patch to the corresponding center value.
            # Reshape current_ids to [B, 1, 1, 1] to broadcast over the [9,9,9] spatial dimensions.
            mask = features_batch != current_ids.view(-1, 1, 1, 1)
            
            # Sum the boolean mask over the spatial dimensions (dimensions 1,2,3)
            # This gives, for each patch, the count of voxels that differ from the center.
            differ_counts = mask.sum(dim=(1, 2, 3))
            
            # Get indices where at least one voxel is different from the center.
            use_i = (differ_counts > 0).nonzero(as_tuple=True)[0]

            if len(use_i) == 1:
                add_i = (use_i + 1) % features_batch.shape[0]  # keep the next index whether or not it has all the same IDs
                use_i = torch.cat([use_i, add_i])  # at least two samples needed for batch normalization when training

            # Pass features through the model with proper flattening
            selected_features = features_batch[use_i,]           # [N, 9, 9, 9]
            flattened_features = selected_features.view(selected_features.size(0), -1)  # [N, 729]
            outputs = self.forward(flattened_features)

            tmp = torch.rand(outputs.shape).to(outputs.device) / 1e12
            switch_i = (outputs + tmp).argmax(1)

            # action_likelyhood += (outputs.sum(0).detach().cpu().reshape((self.act_dim,) * self.num_dims) / batch_output_sz)

            # Find predicted IDs
            next_ids = current_ids.clone()
            # features_batch[use_i] has shape [N, 9, 9, 9]
            flat_features = features_batch[use_i].view(len(use_i), -1)  # Now shape [N, 729]
            # Now, flat_features[torch.arange(len(use_i)), switch_i] yields a tensor of shape [N]
            next_ids[use_i] = flat_features[torch.arange(len(use_i)), switch_i]


            # Calculate loss and accuracy
            if len(use_i > 0):
                # action_likelyhood_true += (labels_batch[use_i,].sum(0).detach().cpu().reshape((self.act_dim,) * self.num_dims) / batch_output_sz)

                # Calculate batch loss
                # Flatten the corresponding labels from [N, 9, 9, 9] to [N, 729]
                selected_labels = labels_batch[use_i]           # shape: [N, 9, 9, 9]
                flattened_labels = selected_labels.view(selected_labels.size(0), -1)  # shape: [N, 729]
                
                # Compute loss with matching shapes
                batch_loss = self.loss_func(outputs, flattened_labels)
                batch_loss.backward()  # Backpropagation per batch
                self.optimizer.step()  # Step optimizer to update weights
                self.optimizer.zero_grad()  # Reset gradients for next batch
                # running_loss += batch_loss
            
        return
      
    def step(self, im_seq, miso_matrix=None, unfold_mem_lim=4e9):
        #"im_seq" can be of shape=[1,1,dim0,dim1,dim2] or a sequence of shape=[num_future, 1, dim0, dim1, dim2]
        #Find the image after "im" given the trained model
        #Also calculates loss and accurate if given an image sequence
        #Calculates misorientation features of "miso_matrix" is given
        
        #Calculate input batch size to maintain memory usage limit 
        num_future = im_seq.shape[0] #number of future steps
        batch_sz = int(unfold_mem_lim/((num_future)*self.act_dim**self.num_dims*self.energy_dim**self.num_dims*64)) #set to highest memory functions - "compute_energy_labels_gen"
        num_iter = int(np.ceil(np.prod(im_seq.shape[1:])/batch_sz))
        
        # Initialize variables and generators
        im = im_seq[0:1,]
        num_future = im_seq.shape[0]-1
        if num_future>0: 
            # the 0 here is self.reg
            labels_gen = fs.compute_labels_gen(im_seq, batch_sz, self.act_dim, self.energy_dim, 0, self.pad_mode) # Labels
            im_next_true_split = im_seq[1:2,].flatten().split(batch_sz)
        
        # needs to be further batched, split into features and train on those batches
        im_unfold_gen = fs.unfold_in_batches(im[0,0], batch_sz, [self.obs_dim,]*self.num_dims, [1,]*self.num_dims, self.pad_mode)
        if miso_matrix is None:
            features_gen = fs.compute_features_gen(im, batch_sz, self.obs_dim, self.pad_mode) # Inputs
        else: 
            features_gen = fs.compute_features_miso_gen(im, batch_sz, miso_matrix, self.obs_dim, self.pad_mode)
        
        # Expect this to be similar to batch_sz
        batch_output_sz = batch_sz # experiment with changing batch_sz as well

        # Find next image
        running_loss = 0
        running_accuracy = 0
        total_batches = 0
        action_likelyhood = torch.zeros((self.act_dim,)*self.num_dims)
        action_likelyhood_true = torch.zeros((self.act_dim,)*self.num_dims)
        
        im_next_log = []
        logx = []
        logy = []
        
        for i in range(num_iter):
            
            # Only use neighborhoods that have more than one ID (have a potential to change ID)
            im_unfold = next(im_unfold_gen).reshape(-1, self.obs_dim**self.num_dims)
            mid_i = int(self.obs_dim**self.num_dims/2)
            current_ids = im_unfold[:,mid_i]
            use_i = (im_unfold != current_ids[:,None]).sum(1).nonzero()[:,0]
            
            if len(use_i)==1: 
                add_i = (use_i+1)%im_unfold.shape[0] #keep the next index whether or not it has all the same IDs
                use_i = torch.cat([use_i, add_i]) #at least two samples needed for batch normalization when training
            
            # Get batch size for this iteration
            batch_output_sz = len(use_i)

            # Pass features through model
            features = next(features_gen).reshape(-1, self.obs_dim**self.num_dims)
            outputs = self.forward(features[use_i,])
            
            tmp = torch.rand(outputs.shape).to(outputs.device)/1e12
            switch_i = (outputs+tmp).argmax(1)
            # switch_i = (outputs).argmax(1)

            action_likelyhood += (outputs.sum(0).detach().cpu().reshape((self.act_dim,)*self.num_dims) / batch_output_sz)
            
            # Find predicted IDs
            next_ids = current_ids.clone()
            next_ids[use_i] = im_unfold[use_i,][torch.arange(len(use_i)),switch_i]
            im_next_log.append(next_ids)
            
            iii, jjj = fs.shape_indices(switch_i, torch.Tensor([action_likelyhood.shape[0], action_likelyhood.shape[1]]))
            logx.append(iii.float().mean().cpu()-action_likelyhood.shape[0] // 2)
            logy.append(jjj.float().mean().cpu()-action_likelyhood.shape[1] // 2)
            
            # Calculate loss and accuracy
            if num_future>0 and batch_output_sz > 0: 
                labels = next(labels_gen).reshape(-1, self.act_dim**self.num_dims)
                if len(use_i>0):
                    action_likelyhood_true += (labels[use_i,].sum(0).detach().cpu().reshape((self.act_dim,)*self.num_dims) / batch_output_sz)
                    
                    # Calculate batch loss
                    batch_loss = self.loss_func(outputs, labels[use_i,])
                    running_loss += batch_loss
                    
                    # Calculate batch accuracy
                    next_ids_true = im_next_true_split[i]
                    batch_accuracy = torch.sum(next_ids[use_i] == next_ids_true[use_i]).float() / batch_output_sz
                    running_accuracy += batch_accuracy
                    
                    total_batches += 1
                    
        # Concatenate batches to form next image (as predicted)
        im_next = torch.cat(im_next_log).reshape(im.shape)
        
        self.logx.append(np.stack(logx).mean(0)) #center of mass of whole step
        self.logy.append(np.stack(logy).mean(0)) #center of mass of whole step
        
        # Find average of loss and accuracy, all need to be reorganized
        if num_future>0 and total_batches>0: 
            action_likelyhood /= total_batches
            action_likelyhood_true /= total_batches
            final_loss = running_loss / total_batches 
            final_accuracy = running_accuracy / total_batches
            return im_next, final_loss, final_accuracy, action_likelyhood, action_likelyhood_true
            
        return im_next
        
        
    def evaluate_model(self):
        
        # self.eval()
        
        with torch.no_grad():
            im_next, loss, accuracy, action_likelyhood, action_likelyhood_true = self.step(self.im_seq, self.miso_matrix)
            self.im_next_val, loss_val, accuracy_val, self.action_likelyhood_val, self.action_likelyhood_true_val = self.step(self.im_seq_val, self.miso_matrix_val)
            self.training_loss.append(loss.item())
            self.training_acc.append(accuracy.item())
            self.validation_loss.append(loss_val.item())
            self.validation_acc.append(accuracy_val.item())
    
        
    def train_model(self):
        # Train the model using the custom-batched train_step
        self.optimizer.zero_grad()  # Zero the gradient
        self.train_step(self.im_seq, self.miso_matrix)
        
        
    def plot(self, fp_results='./plots'):
        
        # Plot the initial image and next images (predicted and true)
        s_3d = (int(self.im_seq_val.shape[-1]/2),)*(self.num_dims-2)
        s0 = (0,0,slice(None),slice(None)) + s_3d
        s1 = (1,0,slice(None),slice(None)) + s_3d
        
        fig, axs = plt.subplots(1,3)
        axs[0].matshow(self.im_seq_val[s0].cpu(), interpolation='none')
        axs[0].set_title('Current')
        axs[0].axis('off')
        axs[1].matshow(self.im_next_val[s0].cpu(), interpolation='none') 
        axs[1].set_title('Predicted Next')
        axs[1].axis('off')
        axs[2].matshow(self.im_seq_val[s1].cpu(), interpolation='none') 
        axs[2].set_title('True Next')
        axs[2].axis('off')
        plt.savefig('%s/sim_vs_true.png'%fp_results)
        plt.show()
        
        #Plot the action distributions (predicted and true)
        s_3d = (int(self.act_dim/2),)*(self.num_dims-2)
        s = (slice(None),slice(None),) + s_3d
        
        
        aaa = self.action_likelyhood_val[s].cpu()
        bbb = (aaa-aaa.mean())/aaa.std()
        
        aaa1 = self.action_likelyhood_true_val[s].cpu()
        bbb1 = (aaa1-aaa1.mean())/aaa1.std()
        
        
        ctr = int((self.act_dim-1)/2)
        fig, axs = plt.subplots(1,2)
        # p1 = axs[0].matshow(self.action_likelyhood_val[s].cpu(), vmin=0, vmax=1, interpolation='none')
        p1 = axs[0].matshow(bbb, vmin=-3, vmax=3, interpolation='none')
        fig.colorbar(p1, ax=axs[0])
        axs[0].plot(ctr,ctr,marker='x')
        axs[0].set_title('Predicted')
        axs[0].axis('off')
        # p2 = axs[1].matshow(self.action_likelyhood_true_val[s].cpu(), vmin=0, vmax=1, interpolation='none') 
        p2 = axs[1].matshow(bbb1, vmin=-3, vmax=3, interpolation='none') 
        fig.colorbar(p2, ax=axs[1])
        axs[1].plot(ctr,ctr,marker='x')
        axs[1].set_title('True')
        axs[1].axis('off')
        plt.savefig('%s/action_likelihood.png'%fp_results)
        plt.show()
        
        #Plot loss and accuracy (training and validation)
        fig, axs = plt.subplots(1,2)
        axs[0].plot(self.validation_loss, '-*', label='Validation')
        axs[0].plot(self.training_loss, '--*', label='Training')
        axs[0].set_title('Loss (%.3f)'%np.min(self.validation_loss))
        axs[0].legend()
        axs[1].plot(self.validation_acc, '-*', label='Validation')
        axs[1].plot(self.training_acc, '--*', label='Training')
        axs[1].set_title('Accuracy (%.3f)'%np.max(self.validation_acc))
        axs[1].legend()
        plt.savefig('%s/train_val_loss_accuracy.png'%fp_results)
        plt.show()
        
        plt.close('all')
        
        
    def save(self, name):
        torch.save(self.state_dict(), name)
    
    
    
def train_primme(trainset, num_eps, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", plot_freq=None, if_miso=False, multi_epoch_safe=False):
    
    with h5py.File(trainset, 'r') as f: dims = len(f['ims_id'].shape)-3
    append_name = trainset.split('_kt')[1]
    modelname = "./data/model_dim(%d)_sz(%d_%d)_lr(%.0e)_reg(%d)_ep(%d)_kt%s"%(dims, obs_dim, act_dim, lr, reg, num_eps, append_name)
    agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode=pad_mode, learning_rate=lr, reg=reg, num_dims=dims, if_miso=if_miso, if_training=True).to(device)
    
    # # Code to split into GPUs
    # top_agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode=pad_mode, learning_rate=lr, reg=reg, num_dims=dims, if_miso=if_miso).to(device)
    # top_agent = nn.DataParallel(top_agent)
    
    # # Move model to device
    # top_agent.to(device)
    # agent = top_agent.module
    
    best_validation_loss = 1e9
    
    for i in tqdm(range(num_eps), desc='Epochs', leave=True):
        agent.sample_data(trainset)
        agent.train_model()
        agent.evaluate_model()
        
        val_loss = agent.validation_loss[-1]
        if multi_epoch_safe and i % 100 ==0:
            agent.save(f"{modelname[:-3]}_at_epoch({i}).h5")

        if val_loss<best_validation_loss:
            best_validation_loss = val_loss
            agent.save(modelname)
            
        if plot_freq is not None: 
            if i%plot_freq==0:
                agent.plot()
                
                tmpx = np.stack(agent.logx).T
                tmpy = np.stack(agent.logy).T
                
                plt.figure()
                plt.plot(tmpx)
                plt.plot(tmpy)
                plt.show()
                # tmp0 = np.stack(agent.log0).T
                # tmp1 = np.stack(agent.log1).T
                
                # # print(np.mean(tmp0))
                # # print(np.mean(tmp1))
                
                # plt.figure()
                # plt.plot(tmp0[0], 'C0-') 
                # plt.plot(tmp0[1], 'C0--') 
                # plt.plot(tmp1[0], 'C1-') 
                # plt.plot(tmp1[1], 'C1--')  
                # plt.legend(['Mean Distribution (x)','Mean Distribution (y)','Mean Index (x)','Mean Index (y)'])
                # plt.xlabel('Number of training iterations')
                # plt.ylabel('Num pixels from (0,0)')
                # plt.show()
    
    return modelname


def run_primme(ic, ea, nsteps, modelname, miso_array=None, pad_mode='circular', plot_freq=None, if_miso=False):
    
    # Setup variables
    d = len(ic.shape)
    # Dimensions
    obs_dim, act_dim = np.array(modelname.split("sz(")[1].split(")_lr(")[0].split("_")).astype(int)
    # Code Pre-Split
    agent = PRIMME(num_dims=d, obs_dim=obs_dim, act_dim=act_dim, if_training=False).to(device)
        
    agent.load_state_dict(torch.load(modelname))
    agent.pad_mode = pad_mode
    im = torch.Tensor(ic).unsqueeze(0).unsqueeze(0).float().to(device)
    if miso_array is None: miso_array = fs.find_misorientation(ea, mem_max=1) 
    miso_matrix = fs.miso_array_to_matrix(torch.from_numpy(miso_array[None,])).to(device)
    size = ic.shape
    ngrain = len(torch.unique(im))
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
    append_name = modelname.split('_kt')[1]
    sz_str = ''.join(['%dx'%i for i in size])[:-1]
    fp_save = './data/primme_sz(%s)_ng(%d)_nsteps(%d)_freq(1)_kt%s'%(sz_str,ngrain,nsteps,append_name)
    
    # Simulate and store in H5
    with h5py.File(fp_save, 'a') as f:
        
        # If file already exists, create another group in the file for this simulaiton
        num_groups = len(f.keys())
        hp_save = 'sim%d'%num_groups
        g = f.create_group(hp_save)
        
        # Save data
        s = list(im.shape); s[0] = nsteps + 1
        dset = g.create_dataset("ims_id", shape=s, dtype=dtype)
        dset2 = g.create_dataset("euler_angles", shape=ea.shape)
        dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
        dset4 = g.create_dataset("miso_matrix", shape=miso_matrix[0].shape)
        dset[0] = im[0].cpu()
        dset2[:] = ea
        dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
        dset4[:] = miso_matrix[0].cpu() #same values as mis0_array, different format
        
        for i in tqdm(range(nsteps), 'Running PRIMME simulation: '):
            
            # Simulate
            if if_miso: im_next = agent.step(im, miso_matrix)
            else: im_next = agent.step(im)
            im = im_next.clone()

            #Store
            dset[i+1,:] = im[0].cpu()
            
            #Plot
            if plot_freq is not None: 
                if i%plot_freq==0:
                    plt.figure()
                    s = (0,0,slice(None), slice(None),) + (int(im.shape[-1]/2),)*(d-2)
                    plt.imshow(im[s].cpu(), interpolation=None) 
                    plt.show()
                        
    return fp_save