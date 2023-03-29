from torchvision.transforms.functional import affine, hflip, vflip
import numpy as np

class augmentation_segmentation_task():
    def __init__(self):
        self.zoom_range = [0.8, 1.2]
        self.shear_angle = [-5, 5]
        
        #flag to compute specific transformations
        self.enable_shear = True
        self.enable_hflip = True
        self.enable_vflip = True
        self.enable_zoom = True
        
    def horizontal_flip(self, image, target):        
        #random horizontal flip
        if np.random.uniform(0, 1) > 0.5:
            image, target = hflip(image), hflip(target)
        return image, target
    
    def vertical_flip(self, image, target):        
        #random vertical flip    
        if np.random.uniform(0, 1) > 0.5:
            image, target = vflip(image), vflip(target)
        return image, target 
    
    def affine_transform(self, image, target, scale=1, angle=0, translate=[0, 0], shear=0):
        image = affine(image, scale=scale, angle=angle, translate=translate, shear=shear)
        target = affine(target, scale=scale, angle=angle, translate=translate, shear=shear)
        return image, target    
    
    def affine_zoom(self, image, target):        
        #random zoom
        if np.random.uniform(0, 1) > 0.5:
            zoom = np.random.uniform(*self.zoom_range)
            image, target = self.affine_transform(image, target, scale=zoom)
        return image, target
    
    def affine_shear(self, image, target):        
        #random shear
        if np.random.uniform(0, 1) > 0.5:
            shear = np.random.uniform(*self.shear_angle)
            image, target = self.affine_transform(image, target, shear=shear)
        return image, target
        
    def run(self, image, target):
        if self.enable_hflip:
            image, target = self.horizontal_flip(image, target)
            
        if self.enable_vflip:
            image, target = self.vertical_flip(image, target)
            
        if self.enable_zoom:
            image, target = self.affine_zoom(image, target)
            
        if self.enable_shear:
            image, target = self.affine_shear(image, target)
        
        return image, target   