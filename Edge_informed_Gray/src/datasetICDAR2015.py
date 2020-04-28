import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import ndimage, signal
from scipy import misc
import skimage.transform 
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
#import pyblur
#import libs_deblur.pyblur
import random
from datetime import datetime


        
class DatasetICDAR2015(Dataset):
    def __init__(self, root_dir, size=[64, 64], set_name='val', sigmaMin=0.5, sigmaMax=2.5, 
                 transform=None, downsampleFactor=1,lowerpath='HR'):
        self.root_dir = root_dir
        self.transform = transform
        self.downsampleFactor = downsampleFactor
        if set_name=='val' or set_name=='train': self.set_name = set_name
        else: self.set_name = 'test'
        random.seed(datetime.now())
        self.current_set_dir = path.join(self.root_dir, self.set_name)             
        self.samplePathHD = []
        self.samplePathLR = []
        self.imageHD = []
        self.imageHR = []
        self.imageLR = []
        self.nameHD = []
        self.sigmaMin, self.sigmaMax = sigmaMin, sigmaMax
        self.size = size        
        self.kernelSIZE = 3
        self.size_true = size
        self.kernelTransform = transforms.Compose(
            [transforms.ToTensor()             
            ]) 
        self.NLarge = 1000
        if set_name=='val':
            #groundtrue
            for sampleFile in sorted(os.listdir(path.join(self.root_dir, 'VAL','HD'))):
                if sampleFile.endswith('.pgm'):
                    pathhd = path.join(self.root_dir, 'VAL','HD', sampleFile)
                    hdimage = self.reagImage(pathhd)
                    lrimage = self.reagImage(pathhd.replace('HD',lowerpath).replace('hd',lowerpath.lower()),scale=True)
                    
                    for x in range(0,hdimage.shape[0],self.size_true[0]):
                        for y in range(0,hdimage.shape[1],self.size_true[1]):
                            if (x+self.size_true[0]) > hdimage.shape[0]:
                                x = hdimage.shape[0] - self.size_true[0]
                            if (y+self.size_true[1]) > hdimage.shape[1]:
                                y = hdimage.shape[1] - self.size_true[1] 
                            
                            im1 = hdimage[x:x+self.size_true[0],y:y+self.size_true[1]]
                            im2 = lrimage[x:x+self.size_true[0],y:y+self.size_true[1]]
                            self.samplePathHD.append(im1)
                            self.samplePathLR.append(im2)
                            self.nameHD.append(pathhd)
            
        elif set_name == 'train':            
            #groundtrue
            for sampleFile in os.listdir(path.join(self.root_dir, 'TRAIN','HD')):
                if sampleFile.endswith('.pgm'):
                    pathhd = path.join(self.root_dir, 'TRAIN','HD', sampleFile)
                    hdimage = self.reagImage(pathhd)
                    lrimage = self.reagImage(pathhd.replace('HD','LR').replace('hd','lr'),scale=True,scaleFactor=4.0)
                    hrimage = self.reagImage(pathhd.replace('HD','HR').replace('hd','hr'),scale=True,scaleFactor=2.0)
                    #self.imageHD.append(hdimage)
                    #self.imageHR.append(hrimage)
                    #self.imageLR.append(lrimage)
                    
                    numimg = int(hdimage.shape[0]/self.size[0])*10 + int(hdimage.shape[1]/self.size[1]) * 400
                    for i in range(numimg):
                        self.nameHD.append(pathhd)
                        x = random.randint(0,max(hdimage.shape[0]-self.size_true[0],0))
                        y = random.randint(0,max(hdimage.shape[1]-self.size_true[1],0))
                            
                        self.samplePathHD.append(hdimage[x:x+self.size_true[0],y:y+self.size_true[1]].copy())
                        if random.uniform(0,1) > 0.5:
                            self.samplePathLR.append(lrimage[x:x+self.size_true[0],y:y+self.size_true[1]].copy())
                        else:
                            self.samplePathLR.append(hrimage[x:x+self.size_true[0],y:y+self.size_true[1]].copy())
            #error data 
            count = 0
            i = 0
            while i < len(self.samplePathHD):
                if len(self.samplePathHD[i]) <=0 or len(self.samplePathLR[i]) <= 0 :
                    count += 1
                    del  self.samplePathHD[i]
                    del  self.samplePathLR[i]
                    del  self.samplePathHR[i]
                elif self.checkSize(i,self.size_true)==False:
                    count +=1
                    del  self.samplePathHD[i]
                    del  self.samplePathLR[i]
                    del  self.samplePathHR[i]
                else:
                    i +=1
            print("error data: ", count)
                       
        else:
            #groundtrue
            for sampleFile in os.listdir(path.join(self.root_dir, 'TEST','HD')):
                if sampleFile.endswith('.pgm'):
                    pathhd = path.join(self.root_dir, 'TEST','HD', sampleFile)
                    self.samplePathHD.append(self.reagImage(pathhd))
                    self.samplePathLR.append(self.reagImage(pathhd.replace('HD',lowerpath).replace('hd',lowerpath.lower()),scale=True))
                    self.nameHD.append(pathhd)
            

        self.current_set_len = len(self.samplePathHD)   
        
    def checkSize(self,index,size):
        return (self.samplePathHD[index].shape[0]== size[0] and self.samplePathHD[index].shape[1]== size[1] \
               and self.samplePathLR[index].shape[0]== size[0] and self.samplePathLR[index].shape[1]== size[1] )            
    def __len__(self):        
        
        return self.current_set_len
        
        
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    def reagImage(self,imgName,scale=False, scaleFactor=2.0):
        img = misc.imread(imgName)
        #img = img.astype(np.float32) # NOT float!!!
        if len(img.shape)>=3:
            img = self.rgb2gray(img)
         
        if(scale):
            img = misc.imresize(img, scaleFactor, 'bicubic')
            img = np.clip(img, 0, 255) 
            #img = img.astype(np.float32) 
        #img = self.modcrop(img, scale=self.downsampleFactor)
        return img
         
    
    def __getitem__(self, idx):
        img = self.samplePathHD[idx]
        imgNoisy = self.samplePathLR[idx]
        if self.transform:
            img = self.transform(img)
            imgNoisy = self.transform(imgNoisy)        
            
                
        return imgNoisy, img ,"",0
        
            
        
    def modcrop(self, image, scale=3):
        """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
        We need to find modulo of height (and width) and scale factor.
        Then, subtract the modulo from height (and width) of original image size.
        There would be no remainder even after scaling operation.
        """
        if len(image.shape) == 3:
            h, w, _ = image.shape
            h = h - np.mod(h, scale)
            w = w - np.mod(w, scale)
            h, w = int(h),int(w)
            image = image[0:h, 0:w, :]
        else:
            h, w = image.shape
            h = h - np.mod(h, scale)
            w = w - np.mod(w, scale)
            h, w = int(h),int(w)
            image = image[0:h, 0:w]
        return image
    
    
    
        

            
