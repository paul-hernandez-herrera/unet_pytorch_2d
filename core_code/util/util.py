import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def imread(filename):
    if Path(filename).suffix in {'.tif', '.tiff'}:
        return tifffile.imread(filename)
        
def imwrite(filename, arr):
    if Path(filename).suffix in {'.tif', '.tiff'}:
        tifffile.imsave(filename, arr) 
        
        
def get_image_file_paths(input_path):
    input_path = Path(input_path)
    
    # Check if input path is a directory or a file
    if input_path.is_dir():
        img_file_paths  = list(input_path.glob('*.tiff')) + list(input_path.glob('*.tif'))
    elif input_path.suffix in ['.tiff', '.tif']:
        img_file_paths  = [input_path]
    else:
        raise ValueError('Input file format not recognized. Currently only tif files can be processed (.tif or .tiff)')
        
    # Check if any image files were found    
    if not img_file_paths:
        raise ValueError("No .tiff or .tif files found in the given path.")
        
    return img_file_paths    

def create_file_in_case_not_exist(folder_path):
    folder_path.mkdir(parents=True, exist_ok=True)
    return

def show_images_from_Dataset(custom_dataset, n_images_to_display=3):
    img, _  = custom_dataset[0]
    n_channels = img.shape[0]

    # Create figure with subplots for each image and its channels
    fig, axs = plt.subplots(n_images_to_display, n_channels+1, figsize=(10, 20), dpi=80)
    for i in range(n_images_to_display):
        img, target = custom_dataset.__getitem__(np.random.randint(0, len(custom_dataset)))
        img = img.numpy()
        target = target.numpy()
        for j in range(n_channels):
            axs[i,j].imshow(np.squeeze(img[j,:,:]), cmap='gray')
            axs[i,j].set_title('Image ' + str(i) + "--- Ch" + str(j))
            axs[i,j].axis('off')
            fig.colorbar(axs[i,j].imshow(np.squeeze(img[j,:,:]), cmap='gray'), ax=axs[i,j])
            
        for ch in range(target.shape[0]):
            target[ch,:,:] = (ch+1)*target[ch,:,:]            
        axs[i,n_channels].imshow(target.max(axis=0), cmap='gray')
        axs[i,n_channels].set_title('Ground truth (target)')
        axs[i,n_channels].axis('off')
        fig.colorbar(axs[i,n_channels].imshow(target.max(axis=0), cmap='gray'), ax=axs[i,n_channels])
    plt.tight_layout()
    plt.show()       