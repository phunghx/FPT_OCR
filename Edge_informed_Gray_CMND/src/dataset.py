import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as Ff
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from src.datasetCMNDv2 import *
import skimage 
from skimage.filters import threshold_yen

path_to_root = '/root/FPT_OCR/dataCMND32/'


cropSize = [32, 32]
sigmaMin=0.5
sigmaMax=2
batch_size=16

whole_datasets = {set_name: 
                  DatasetCMND(root_dir=path_to_root,
                                     size=cropSize, set_name=set_name, 
                                     transform=None, 
                                     sigmaMin=sigmaMin, sigmaMax=sigmaMax,
                                   downsampleFactor=4.0)
                  for set_name in ['train', 'val']}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type, sigma, scale, hr_size,lr_flist=None, augment=True):
        super().__init__()
        self.augment = augment
        self.dataset_type = dataset_type
        if self.dataset_type != 'test':
           self.lr_data_path = whole_datasets[self.dataset_type].samplePathLR
           self.hr_data_path = whole_datasets[self.dataset_type].samplePathHD
           #self.hr_images = whole_datasets[self.dataset_type].sampleImageHD
           self.dataPatch = whole_datasets[self.dataset_type].dataPatch
        else:
           self.lr_data_path = self.load_flist(lr_flist)

        self.hr_size = hr_size
        self.sigma = sigma
        self.scale = scale

    def __len__(self):
        if self.dataset_type != 'test':
            return len(self.hr_data_path)
        else:  return len(self.lr_data_path)

    def __getitem__(self, index):
        #try:
        if self.dataset_type != 'test':
            item = self.load_item(index)
        else:
            item = self.load_item_test(index)
        #except:
        #    print('loading error: ' + self.hr_data[index])
        #    item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.lr_data_path[index]
        return os.path.basename(name)
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
    def load_item_test(self, index):

        lr_img = scipy.misc.imread(self.lr_data_path[index])
        #import pdb;pdb.set_trace()
        if len(lr_img.shape)>=3:
            lr_img = self.rgb2gray(lr_img)
        imgh, imgw = lr_img.shape[0:2]
        
        hr_img = lr_img.copy()
        #import pdb;pdb.set_trace()
        hr_edge = self.load_edge(hr_img,False)
        lr_edge = self.load_edge(lr_img,True)
                    
        # crop to 64,64    
        
        return self.to_tensor(lr_img), self.to_tensor(hr_img), self.to_tensor(lr_edge), self.to_tensor(hr_edge)
        
    def load_item(self, index):

        size = self.hr_size
        scale = self.scale
        flag = True
        size_patch, scaleFactor = self.dataPatch[index]
        hr_img = scipy.misc.imread(self.hr_data_path[index])
        #if len(hr_img.shape)>=3:
        #    hr_img = self.rgb2gray(hr_img)
        imgh, imgw = hr_img.shape[0:2]
        '''
        hw = 16
        if self.dataset_type=='train':
            i = random.randint(0, imgh - hw)
            j = random.randint(0, imgw - hw)
            hr_img = hr_img[i:i+hw, j:j+hw]
        
            imgh, imgw = hr_img.shape[0:2]
        '''
        lr_img = scipy.misc.imresize(hr_img.copy(), [imgh // scaleFactor, imgw // scaleFactor])
        lr_img = scipy.misc.imresize(lr_img, [imgh,imgw])
        #import pdb;pdb.set_trace()
        hr_edge = self.load_edge(hr_img,False)
        lr_edge = self.load_edge(lr_img,True)
                    
        # crop to 64,64
        
        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            hr_img = hr_img[:, ::-1, ...]
            lr_img = lr_img[:, ::-1, ...]
            hr_edge = hr_edge[:, ::-1, ...]
            lr_edge = lr_edge[:, ::-1, ...]
            
        
        return self.to_tensor(lr_img), self.to_tensor(hr_img), self.to_tensor(lr_edge), self.to_tensor(hr_edge)

    def load_edge(self, img, low_rate=False):
        #if (low_rate==False):
        #    img_blur = skimage.filters.gaussian(img, sigma=1)
        #    thresh = threshold_yen(img_blur)
        #    binary = img_blur > thresh
        #else:
        #    thresh = threshold_yen(img)
        #    binary = img > thresh
         
        canny_img = canny(img.astype(np.float), sigma=self.sigma)
        return canny_img.astype(np.float)
        #return np.bitwise_and(canny_img , binary).astype(np.float)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        
        img_t = Ff.to_tensor(img).float()
        
        return img_t

    def resize(self, img, height, width):
        imgh, imgw = img.shape[0:2]

        if imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
