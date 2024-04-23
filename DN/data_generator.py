# -*- coding: utf-8 -*-

# =============================================================================
# Based on the code from https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch (by Kai Zhang)
# =============================================================================

import pandas as pd
import pdb
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision
import os
import utils
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import math as m
#import torch
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
class DenoisingDataset(Dataset):

    def __init__(self, xs, sigma,img_size=180,random_corp=False):
        super(DenoisingDataset, self).__init__()
        assert sigma<=100 and sigma>=10, 'Sigma was expected between with 10 to 100, but got [{0}]'.format(sigma)
        self.xs = xs
        self.sigma = sigma
        self.img_size =img_size
        self.random_corp = random_corp
    def __getitem__(self, index):
        if self.random_corp:
            batch_x = self.get_patch(self.xs[index]) # augmantation 
            noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
            batch_y = batch_x + noise
        else:
            batch_x = self.xs[index]
            noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
            batch_y = batch_x + noise
        return batch_y, batch_x

    def get_patch(self,img_in):
        _,ih, iw = img_in.size()
        ix = torch.randint(0,iw - self.img_size + 1,(1,)).item()
        iy = torch.randint(0,ih - self.img_size + 1,(1,)).item()
        def _augment(img):
            hflip =  np.random.random() < 0.5
            vflip =  np.random.random() < 0.5
            if hflip: img = torch.flip(img,(2,))
            if vflip: img =  torch.flip(img,(1,))
            return img
        img_in = img_in[:,iy:iy + self.img_size, ix:ix + self.img_size]

        return _augment(img_in)


    def collate_fn(self, batch):
        batch_y, batch_x = list(zip(*batch))
        batch_x = torch.stack([img for img in batch_x])
        batch_y = torch.stack([img for img in batch_y])
        return batch_y, batch_x
    def __len__(self):
        return self.xs.size(0)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPEG"])


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def set_image(img,size):# Crop image to arange image size
    if img.shape[0] != img.shape[1]:
        if img.shape[0] <img.shape[1]:
            y,x ,_= img.shape
            x= y
            img= img[:,0:x,:]
            width =img.shape[0]
        else:
            y,x ,_ = img.shape
            y=x
            img= img[0:y,:,:]
            width =img.shape[1]
    else: width =img.shape[0]
    if size is not None:

        img = crop_center(size,size)
        if len(img.shape)<3:
            img = np.expand_dims(img, axis=2)

    return img




def gen_patches(file_name,patch_size,patch_crop,stride,large_size=False,scales=[1],color=1):
    if color == 1:
        img = cv2.imread(file_name, 0)  # gray scale
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif color == 3:
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)  # BGR or G
    if large_size:
        patch_size=large_size
    patches = []
    if patch_crop==True:
        w, h,_ = img.shape
        for i in range(0, w-patch_size+1, stride):
            for j in range(0, h-patch_size+1, stride):
                x = img[i:i+patch_size, j:j+patch_size,:]
                patches.append(x)

    else:
        patches.append(set_image(img,patch_size))
    return patches


def nsat_datagenerator(root = "dataset/",classes = ['mix'], verbose=False, batch_size=32, patch_size=63,
            Nsat=400,patch_crop=False,large_size=False, stride = 100,scales=[1],color=1):
    # generate clean patches from a dataset
    data = []
    for index,c in enumerate(classes):
        # index=1
        file_list = glob.glob(root+c+'/*')  # get name list 
        if Nsat[1] == -1:
            file_list=file_list
        else:
            file_list=file_list[Nsat[0]:Nsat[1]]
        for i in range(len(file_list)):
            assert is_image_file(file_list[i]),"[{0}] is not image file. check your dataset".format(file_list[i])
            patches = gen_patches(file_list[i],patch_size,patch_crop,stride,large_size=large_size,scales=scales,color=color)
            for patch in patches:    

                data.append(patch)


    data = np.array(data, dtype='uint8')

    print(data.shape)



    print('^_^-training data finished-^_^')
    return data

def Get_test_set(test_set=[],root='/',hyper_params=[]):
        np.random.seed(seed=hyper_params.seed)
        file_lists=[] # load test set for val  
        dataset = [] 
        print('-------------------')
        for set_cur in test_set:
            print('SET:'+set_cur)
            file_list = sorted(os.listdir(os.path.join(root , set_cur)))
            file_lists.append(file_list) # shape -> [file_list,file_list,file_list]
        for label_index,(set_cur,file_list) in enumerate(zip(test_set,file_lists)):
            data_arr=[]
            for im in file_list:
                if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                    if hyper_params['color'] == 1:
                        org_img =  cv2.imread(os.path.join(root ,set_cur, im), 0)
                    elif hyper_params['color'] == 3:
                        org_img = cv2.imread(os.path.join(root ,set_cur, im), cv2.IMREAD_UNCHANGED) # BGR or G
                    if hyper_params['handle_test_size'] :
                        x,x_w_pad_size,x_h_pad_size = utils.pad_to_image(org_img,hyper_params['test_mod_size'] )
                    else:
                        x = org_img
                    x =x.astype(np.float32)/255.0
                    y = x + np.random.normal(0, hyper_params.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                    y = y.astype(np.float32)
                    if y.ndim == 3:
                        y = np.transpose(y, (2,0, 1))
                        y_ = torch.from_numpy(y).unsqueeze(0)
                    else:
                        y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
                    #pdb.set_trace()
                    temp_set = np.array([org_img, y_ , x_w_pad_size, x_h_pad_size], dtype = object)
                    #temp_set = [org_img, y_ , x_w_pad_size, x_h_pad_size]
                    data_arr.append(temp_set) # 
            #dataset.append(np.array([set_cur,np.array(data_arr)]))
            dataset.append([set_cur, data_arr])
        print('-------------------')
        return dataset



class DatasetSR(Dataset):
    '''
    Made for RGB 0-255
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['color'] 
        self.sf = opt['scale']
        self.patch_size = self.opt['imgSize'] 
        self.test_set = self.opt['test_set'] 
        self.L_size = self.patch_size // self.sf
        print(self.L_size)

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        if len(opt['train_set'] )> 1:
            raise NotImplementedError 
        self.paths_H = utils.get_image_paths(opt['train_root']+'/'+opt['train_set'][0])
        # else:
        #     self.file_lists=[] # load test set for val  
        #     print('-------------------')
        #     for set_cur in self.test_set:
        #         print('SET:'+set_cur)
        #         file_list = sorted(os.listdir(os.path.join(opt[phase+'_root'] , set_cur)))
        #         self.file_lists.append(file_list) # shape -> [file_list,file_list,file_list]
        #     self.paths_H=self.file_lists
            
    def __getitem__(self, index):
        # start_time = time.time()
        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = utils.imread_uint(H_path, self.n_channels)
        # img_H = utils.uint2single(img_H)
        # print('read time ', time.time()-start_time)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = utils.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        # --------------------------------
        # sythesize L image via matlab's bicubic
        # --------------------------------
        H, W = img_H.shape[:2]
        img_L = np.uint8(utils.imresize_np(img_H, 1 / self.sf, True))
        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------

        H, W, C = img_L.shape

        # --------------------------------
        # randomly crop the L patch
        # --------------------------------

        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
        # --------------------------------
        # crop corresponding H patch
        # --------------------------------
        rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
        img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

        # --------------------------------
        # augmentation - flip and/or rotate
        # --------------------------------
        mode = np.random.randint(0, 8)
        img_L, img_H = utils.augment_img(img_L, mode=mode), utils.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = utils.uint2int_tensor3(img_H), utils.uint2int_tensor3(img_L)


        # if L_path is None:
        #     L_path = H_path

        return img_L ,img_H
        # else:
        #     dataset=[]
        #     for label_index,(set_cur,file_list) in enumerate(zip(self.test_set,self.file_lists)):
        #         data_arr=[]
        #         for im in file_list:
        #             im=os.path.join(self.opt[self.phase+'_root'] , set_cur,im)
        #             # ------------------------------------
        #             # get H image
        #             # ------------------------------------
        #             img_H = utils.imread_uint(im, self.n_channels)
        #             img_H_padded,x_w_pad_size,x_h_pad_size = utils.pad_to_image(img_H,self.opt['test_mod_size'] )
        #             # ------------------------------------
        #             # get L image
        #             # ------------------------------------
        #             # --------------------------------
        #             # sythesize L image via matlab's bicubic
        #             # --------------------------------
        #             H, W = img_H_padded.shape[:2]
        #             img_L = utils.imresize_np(img_H_padded, 1 / self.sf, True)
        #             # ------------------------------------
        #             # L/H pairs, HWC to CHW, numpy to tensor
        #             # ------------------------------------
        #             img_L = utils.uint2tensor4(img_L)

        #             temp_set = np.array([img_H, img_L , x_w_pad_size, x_h_pad_size])
        #             data_arr.append(temp_set) # 

        #         dataset.append(np.array([set_cur,np.array(data_arr)]))
        #     print('-------------------')
        #     return dataset

    def __len__(self):
        return len(self.paths_H)


def Get_test_set_SR(opt, phase='test'):
        n_channels = opt['color'] 
        sf = opt['scale']
        patch_size = opt['imgSize'] 
        test_set = opt[phase+'_set'] 
        file_lists=[] # load test set for val  
        print('-------------------')
        for set_cur in test_set:
            print('SET:'+set_cur)
            file_list = sorted(os.listdir(os.path.join(opt[phase+'_root'] , set_cur)))
            file_lists.append(file_list) # shape -> [file_list,file_list,file_list]
        paths_H=file_lists
        dataset=[]
        for label_index,(set_cur,file_list) in enumerate(zip(test_set,file_lists)):
            data_arr=[]
            for im in file_list:
                im=os.path.join(opt[phase+'_root'] , set_cur,im)
                # ------------------------------------
                # get H image
                # ------------------------------------
                img_H = utils.imread_uint(im, n_channels)
                img_H_padded,x_w_pad_size,x_h_pad_size = utils.pad_to_image(img_H,opt['test_mod_size'] )
                # ------------------------------------
                # get L image
                # ------------------------------------
                # --------------------------------
                # sythesize L image via matlab's bicubic
                # --------------------------------
                H, W = img_H_padded.shape[:2]
                img_L = utils.imresize_np(img_H_padded, 1 / sf, True)

                # ------------------------------------
                # L/H pairs, HWC to CHW, numpy to tensor
                # ------------------------------------
                img_L = utils.uint2int_tensor4(img_L)

                temp_set = np.array([img_H, img_L , x_w_pad_size, x_h_pad_size])
                data_arr.append(temp_set) # 

            dataset.append(np.array([set_cur,np.array(data_arr)]))
        print('-------------------')
        return dataset
class Blind_DatasetCash(Dataset):
    '''
    Made for RGB 0-255
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt,sigma):
        super(Blind_DatasetCash, self).__init__()
        # options
        self.opt = opt
        self.n_channels = opt['color'] 
        self.patch_size = self.opt['imgSize'] 
        self.test_set = self.opt['test_set'] 
        self.rgb_range = self.opt['rgb_range'] 
        self.sigma_max=sigma
        self.sigma_min=sigma
        if len(opt['train_set'] )> 1:
            raise NotImplementedError 
        # image path
        self.paths_H = np.array(utils.get_image_paths(opt['train_root']+'/'+opt['train_set'][0]+'/'))
        self.n_images = len(self.paths_H)
        # settings for cash
        self.cash=None # cash
        self.divide=4
        self.im_num_slice  = self.n_images//self.divide
        n_patches =56 * 1000# exept total images
        # n_patches =16 * 1000# exept total images

        print('n_patches:',n_patches)
        self.repeat = max(n_patches // self.n_images, 1) # if not enogh exept total images, repeat dataset
        self.cashpath=opt['train_root']+'/'+opt['train_set'][0]+'_cash/' # path of cash
        self.last =len(self.paths_H[self.im_num_slice*(self.divide-1):]) # calc last index of images
        # shuffle images of each slice
        if not os.path.exists(self.cashpath):
            os.makedirs(self.cashpath)
        self.cash_rands =[np.random.permutation(self.im_num_slice),np.random.permutation(self.im_num_slice),\
            np.random.permutation(self.im_num_slice),np.random.permutation(self.last)]
        print("h############################")
        print('trani',self.n_images)
        print("h############################")

        self.save_cash() # save cash to npy
        self.last_index=0 # last cashd index 
        self.get_cash(0) # load npy to self.cash

    def save_cash(self):
        # load images and save 
        indexrand = np.random.permutation(self.n_images)
        self.paths_H = self.paths_H[indexrand] #shufle
        for now_index in range(self.divide):
            cash =[]
            start= now_index*self.im_num_slice
            end= (now_index+1)*self.im_num_slice
            if now_index==3:end= self.n_images # for out of bound
            
            for HR_path in tqdm(self.paths_H[start:end]):
                img_H = utils.imread_uint(HR_path, self.n_channels)
                cash.append(img_H)
            #pdb.set_trace()    
            np.save(self.cashpath+"cash"+str(now_index)+".npy", np.array(cash, dtype=object))
            #pd.to_pickle(cash, self.cashpath+"cash"+str(now_index) + ".pkl")

    def get_cash(self,now_index):
        self.cash =np.load(self.cashpath+"cash"+str(now_index)+".npy", allow_pickle=True)

    def __getitem__(self, index):
        # resolve index 
        index =  index % len(self.paths_H)
        now_index =m.floor(index/self.im_num_slice)
        if self.last_index!=now_index: # if index is out bound of now cash, load new cash
            self.last_index=now_index
            self.get_cash(self.last_index)
        last_index = self.im_num_slice*now_index
        index=index-last_index
        # load img from cash 
        index = self.cash_rands[now_index][index] # get shufled index
        img_H = self.cash[index]
        H, W, C = img_H.shape


        # --------------------------------
        # randomly crop  
        # --------------------------------
        rnd_h = torch.randint(0, max(0, H-self.patch_size),(1,)).item()
        rnd_w = torch.randint(0, max(0, W-self.patch_size),(1,)).item()
        img_H = img_H[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]

        # --------------------------------
        # augmentation - flip and/or rotate
        # --------------------------------
        mode = torch.randint(0, 8,(1,)).item()
        img_H = utils.augment_img(img_H, mode=mode)
        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        if self.rgb_range==255:
            img_H= utils.uint2int_tensor3(img_H)
        else:
            img_H = utils.uint2tensor3(img_H)
        img_L=img_H.clone()#.to(torch.device("cuda:0"))
        # noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/float(self.rgb_range)
        # noise_level = torch.FloatTensor([self.sigma_max])/float(self.rgb_range)


        # ---------------------------------
        # add noise
        # ---------------------------------
        # noise = torch.randn(img_L.size()).mul_(noise_level).float()
        # img_L.add_(noise)
        noise = torch.randn(img_L.size()).mul_(self.sigma_max/255.0)#.to(torch.device("cuda:0"))
        img_L = img_L + noise
        return img_L ,img_H

    def __len__(self):
        return len(self.paths_H)*self.repeat


class RealtimeLoaddataset(Dataset):
    '''
    Made for RGB 0-255
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt,sigma):
        super(RealtimeLoaddataset, self).__init__()
        # options
        self.opt = opt
        self.n_channels = opt['color'] 
        self.patch_size = self.opt['imgSize'] 
        self.test_set = self.opt['test_set'] 
        self.rgb_range = self.opt['rgb_range'] 
        self.sigma_max=sigma
        self.sigma_min=sigma
        if len(opt['train_set'] )> 1:
            raise NotImplementedError 
        # image path
        self.paths_H = np.array(utils.get_image_paths(opt['train_root']+'/'+opt['train_set'][0]+'/'))
        self.n_images = len(self.paths_H)
        # settings for cash
        self.cash=None # cash
        self.divide=4
        self.im_num_slice  = self.n_images//self.divide
        # n_patches =56 * 1000# exept total images
        n_patches =1 * 1000# exept total images
        self.repeat = max(n_patches // self.n_images, 1) # if not enogh exept total images, repeat dataset
        # shuffle images of each slice

        print("h############################")
        print('trani',self.n_images)
        print("h############################")



    def __getitem__(self, index):
        # resolve index 
        index =  index % len(self.paths_H)
        H_path = self.paths_H[index]
        img_H = utils.imread_uint(H_path, self.n_channels)
        H, W, C = img_H.shape


        # --------------------------------
        # randomly crop  
        # --------------------------------
        rnd_h = torch.randint(0, max(0, H-self.patch_size),(1,)).item()
        rnd_w = torch.randint(0, max(0, W-self.patch_size),(1,)).item()
        img_H = img_H[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]

        # --------------------------------
        # augmentation - flip and/or rotate
        # --------------------------------
        mode = torch.randint(0, 8,(1,)).item()
        img_H = utils.augment_img(img_H, mode=mode)
        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        if self.rgb_range==255:
            img_H= utils.uint2int_tensor3(img_H)
        else:
            img_H = utils.uint2tensor3(img_H)
        img_L=img_H.clone()
        # noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/float(self.rgb_range)
        # noise_level = torch.FloatTensor([self.sigma_max])/float(self.rgb_range)


        # ---------------------------------
        # add noise
        # ---------------------------------
        # noise = torch.randn(img_L.size()).mul_(noise_level).float()
        # img_L.add_(noise)
        noise = torch.randn(img_L.size()).mul_(self.sigma_max/255.0)
        img_L = img_L + noise
        return img_L ,img_H

    def __len__(self):
        return len(self.paths_H)*self.repeat

# from omegaconf import OmegaConf

# # # Get_test_set(['Set12','Set68','Urban100'],'Dataset/test/',OmegaConf.load('./confs/GTCNN.yaml').experiment)
# # # Get_test_set(['Set12','Set68','Urban100'],'Dataset/test/',OmegaConf.load('./confs/GTCNN.yaml').experiment)
# from torch.utils.data import DataLoader
# DatasetSR = DatasetSR(OmegaConf.load('./confs/GTCNN_SR.yaml').dataset)
# # dataloader  = DataLoader(dataset=DatasetSR, num_workers=1,drop_last=True, batch_size=50, pin_memory=True,shuffle=True)
# # Get_test_set_SR = Get_test_set_SR(OmegaConf.load('./confs/GTCNN_SR.yaml').dataset,phase='test')
# print(DatasetSR[0][0])
# # for i in Get_test_set_SR:

# #     print(i[0])
#     print(i[1])
#     break
# # for i,u in DatasetSR:
# #     print(i.shape)
# #     break