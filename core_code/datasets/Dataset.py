from ..util import util
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torch import tensor
from ..util.preprocess import preprocess_image

class CustomImageDataset(Dataset):
    """
    The purpose of this class is to load input and target image pairs from two separate directories, 
    preprocess the input image if required, and return them as a tuple of PyTorch tensors.    
    """
    def __init__(self, folder_input, folder_target, enable_preprocess = False):
        valid_suffix = {".tif", ".tiff"}
        
        #saving the variables
        self.folder_input = Path(folder_input)
        self.folder_target = Path(folder_target)
        self.enable_preprocess = enable_preprocess
        
        self.data_augmentation_flag = False
        
        #reading all files in target folder:
        self.file_names = [p.name for p in Path(self.folder_input).iterdir() if p.suffix in valid_suffix] 

        check_trainingset_file_matching(self.file_names, self.folder_target)
        
        
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        #reading input and target images
        input_img = util.imread(Path(self.folder_input, self.file_names[idx]))
        target_img = util.imread(Path(self.folder_target, self.file_names[idx]))
        
        #preprocess image if required
        if self.enable_preprocess:
            input_img = preprocess_image(input_img)
            
        #converting numpy to tensor
        input_img = tensor(input_img.astype(np.float32)).float()
        target_img = tensor(target_img.astype(np.uint8))


        #U-Net requires dimensions to be [C,W,H]. Make sure that we have Channel dimension
        input_img = input_img.unsqueeze(0) if input_img.dim() == 2 else input_img
        target_img = target_img.unsqueeze(0) if target_img.dim() == 2 else target_img
            
        
        if self.data_augmentation_flag:
            input_img, target_img = self.data_augmentation_object.run(input_img, target_img)
        
        
        return input_img, target_img 


    def set_data_augmentation(self, augmentation_flag = False, data_augmentation_object = None):
        """
        this method is used to set a data augmentation flag and object. 
        The data_augmentation_flag is a boolean indicating whether data augmentation should be performed or not
        data_augmentation_object is an object containing the data augmentation methods to be applied.
        """
        
        self.data_augmentation_flag = augmentation_flag
        self.data_augmentation_object = data_augmentation_object
        if not(augmentation_flag):
            self.data_augmentation_object = None

def check_trainingset_file_matching(file_names, folder_target):
    #verify each image in input_folder has an associated image in the target folder
    missing_files = [f for f in file_names if not Path(folder_target, f).is_file()]
    if missing_files:
        raise ValueError('Missing traces for images: ' + ', '.join(missing_files))        
        
        
        