from .. import util
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torch import tensor
from ..util.preprocess import preprocess_image

class CustomImageDataset(Dataset):
    def __init__(self, folder_input, folder_target, enable_preprocess = False):
        #saving the variables
        self.folder_input = folder_input
        self.folder_target = folder_target
        
        self.enable_preprocess = enable_preprocess
        
        self.data_augmentation_flag = False
        
        #reading all files in target folder:
        self.file_names = [item.name for item in Path(folder_input).iterdir() if ((item.is_file() & (item.suffix=='.tif') | (item.suffix=='.tiff')))] 
        
        
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        #reading input image as tensor float (input) and uint8 (labels)
        input_img = util.imread(Path(self.folder_input, self.file_names[idx]))
        if self.enable_preprocess:
            input_img = preprocess_image(input_img)
        input_img = tensor(input_img.astype(np.float32)).float()
        target_img = tensor(util.imread(Path(self.folder_target, self.file_names[idx])).astype(np.uint8) )


        #U-Net requires dimensions to be [C,W,H]. Make sure that we have Channel dimension
        if len(input_img.shape)==2:
            input_img = input_img[None]
        if len(target_img.shape)==2:
            target_img = target_img[None]
            
        
        if self.data_augmentation_flag:
            input_img, target_img = self.data_augmentation_object.run(input_img, target_img)
        
        
        return input_img, target_img 


    def set_data_augmentation(self, augmentation_flag = False, data_augmentation_object = None):
        self.data_augmentation_flag = augmentation_flag
        self.data_augmentation_object = data_augmentation_object
        
        
        