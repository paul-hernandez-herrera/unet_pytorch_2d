import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def imread(filename):
    ext = Path(filename).suffix
    if ext== '.tif' or ext=='.tiff':
        return tifffile.imread(filename)
        
def imwrite(filename, arr):
    ext = Path(filename).suffix
    if ext== '.tif' or ext=='.tiff':
        tifffile.imsave(filename, arr) 
        
        
def get_image_file_names(file_path):
    file_path = Path(file_path)
    ext = file_path.suffix
    img_file_path = []
    if file_path.is_dir():
        img_file_path.extend(file_path.glob('*.tif'))
        img_file_path.extend(file_path.glob('*.tiff'))
    elif ext=='.tif' or ext=='.tiff':
        img_file_path = [file_path]
    else:
        raise ValueError('Input file format not recognized. Currently only tif files can be processed (.tif or .tiff)')
    return img_file_path   

def create_file_in_case_not_exist(folder_path):
    folder_path.mkdir(parents=True, exist_ok=True)
    return

def display_images_from_Dataset(custom_dataset, n_images_to_display=3):
    img, _  = custom_dataset.__getitem__(0)
    n_channels = img.shape[0]

    plt.figure(figsize=(10, 20), dpi=80)
    k = 0
    for index in np.random.randint(0, len(custom_dataset), size = n_images_to_display):
        img, target = custom_dataset.__getitem__(index)
        img = img.numpy()
        target = target.numpy()
        for i in range(0, n_channels):
            k = k+1
            plt.subplot(n_images_to_display, n_channels+1, k)
            plt.imshow(np.squeeze(img[i,:,:]), cmap='gray')
            plt.colorbar()
            plt.title('Image ' + str(index) + "--- Ch" + str(i))
        k = k+1    
        plt.subplot(n_images_to_display, n_channels+1, k)
        for ch  in range(0, target.shape[0]):
            target[ch,:,:] = (ch+1)*target[ch,:,:]
        plt.imshow(target.max(axis=0), cmap='gray')
        plt.colorbar()
        plt.title('Ground truth (target)') 