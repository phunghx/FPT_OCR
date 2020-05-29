from datasetCMNDPrepare import *
path_to_root = '/root/FPT_OCR/dataCMND/'
path_to_train = '/root/FPT_OCR/dataCMND128LR/'

cropSize = [256, 256]
sigmaMin=0.5
sigmaMax=2
batch_size=16

val_data =DatasetCMND(root_dir=path_to_root, save_output=path_to_train,
                                     size=cropSize, set_name='val', 
                                     transform=None, 
                                     sigmaMin=sigmaMin, sigmaMax=sigmaMax,
                                   downsampleFactor=4.0)
train_data =DatasetCMND(root_dir=path_to_root, save_output=path_to_train,
                                     size=cropSize, set_name='train',                                                                                                                                                 transform=None,
                                     sigmaMin=sigmaMin, sigmaMax=sigmaMax,
                                   downsampleFactor=4.0)
                                   
                                   
