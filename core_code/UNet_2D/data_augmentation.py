from torchvision.transforms.functional import affine, hflip, vflip
import numpy as np

class augmentation_segmentation_task():
    def __init__(self):
        #nothing to initialize
        self.zoom_range = [0.8, 1.2]
        self.shear_angle = [-5, 5]
        
        #flag to compute specific transformations
        self.shear_flag = True
        self.hflip_flag = True
        self.vflip_flag = True
        self.zoom_flag = True
        return
        
    def horizontal_flip(self, image, target):        
        #random horizontal flip
        if np.random.uniform(0, 1) > 0.5:
            image = hflip(image)
            target = hflip(target)
        return image, target
    
    def vertical_flip(self, image, target):        
        #random vertical flip    
        if np.random.uniform(0, 1) > 0.5:
            image = vflip(image)
            target = vflip(target)
        return image, target 
    
    def affine_zoom(self, image, target):        
        #random zoom
        if np.random.uniform(0, 1) > 0.5:
            zoom = np.random.uniform(low= self.zoom_range[0], high= self.zoom_range[1])
            image = affine(image, scale = zoom, angle = 0, translate = [0,0], shear = 0)
            target = affine(target, scale = zoom, angle = 0, translate = [0,0], shear = 0)
        return image, target
    
    def affine_shear(self, image, target):        
        #random shear
        if np.random.uniform(0, 1) > 0.5:
            shear = np.random.uniform(low= self.shear_angle[0], high= self.shear_angle[1])
            image = affine(image, scale = 1, angle = 0, translate = [0,0], shear = shear)
            target = affine(target, scale = 1, angle = 0, translate = [0,0], shear = shear)
        return image, target
        
    def run(self, image, target):
        if self.hflip_flag:
            image, target = self.horizontal_flip(image, target)
            
        if self.vflip_flag:
            image, target = self.vertical_flip(image, target)
            
        if self.zoom_flag:
            image, target = self.affine_zoom(image, target)
            
        if self.shear_flag:
            image, target = self.affine_shear(image, target)
        
        return image, target   