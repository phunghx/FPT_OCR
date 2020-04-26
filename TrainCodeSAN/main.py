import torch

import utility
#import data
import model
import loss
from option import args
from trainer import Trainer
from datasetICDAR2015 import *

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

path_to_root = '/data1/phung/OCR/ICDAR2015/DATA/'
transform4Image = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((127.,127.,127.),(127.,127.,127.)) # (mean, std)
    ]) # (mean, std)

cropSize = [32, 32]
sigmaMin=0.5
sigmaMax=2
batch_size=16

whole_datasets = {set_name: 
                  DatasetICDAR2015(root_dir=path_to_root,
                                     size=cropSize, set_name=set_name, 
                                     transform=transform4Image, 
                                     sigmaMin=sigmaMin, sigmaMax=sigmaMax,
                                   downsampleFactor=4.0,lowerpath='LR')
                  for set_name in ['train', 'val']}


dataloaders = {set_name: DataLoader(whole_datasets[set_name], 
                                    batch_size=batch_size,
                                    shuffle=set_name=='train', 
                                    num_workers=8) # num_work can be set to batch_size
               for set_name in ['train', 'val']}

if checkpoint.ok:
    #loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, dataloaders, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

