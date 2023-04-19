import matplotlib.pyplot as plt
import numpy as np
from ..parameters_interface.ipwidget_basic import set_dropdown
from ipywidgets import widgets
from core_code.parameters_interface.ipwidget_basic import set_IntSlider, set_checkbox
import napari
from IPython.display import display
from .util import imread
from pathlib import Path

########################################################################################################
########################################################################################################

def show_images_from_Dataset(custom_dataset, n_images_to_display=3):
    # Create figure with subplots for each image and its channels
    n_channels = custom_dataset[0][0].shape[0]
    fig, axs = plt.subplots(n_images_to_display, n_channels+1, figsize=(10, 20), dpi=80)
    
    for i in range(n_images_to_display):
        img, target = custom_dataset[np.random.randint(len(custom_dataset))]
        
        img, target = np.asarray(img), np.asarray(target)
        img, target = convert_img_shape_to_C_W_H(img, target)
        
        for j in range(n_channels):
            axs[i,j].imshow(img[j], cmap='gray')
            axs[i,j].set_title(f'Image {i} - Ch{j}')
            axs[i,j].axis('off')
            fig.colorbar(axs[i,j].imshow(img[j], cmap='gray'), ax=axs[i,j])
            
        target = np.multiply(target, np.arange(1, n_channels+1).reshape(-1, 1, 1))
        axs[i, n_channels].imshow(np.max(target, axis=0), cmap='gray')
        axs[i, n_channels].set_title('Ground truth (target)')
        axs[i, n_channels].axis('off')
        fig.colorbar(axs[i, n_channels].imshow(np.max(target, axis=0), cmap='gray'), ax=axs[i, n_channels])
        
    plt.tight_layout()
    plt.show()    
    
########################################################################################################
########################################################################################################
    
def convert_img_shape_to_C_W_H(img, target):
    if img.ndim == 4:
        img = np.max(img, axis=1)
        target = np.max(target, axis=1)
    return img, target     
    
########################################################################################################
########################################################################################################

def show_images_side_by_side_interactive(left_image_paths, right_image_paths):
    global main_container
    main_container = widgets.HBox()
    
    dropdown_options = [(Path(file_name).name, str(idx)) for idx, file_name in enumerate(left_image_paths)] 
    
    dropdown_w = set_dropdown('Image to show: ', dropdown_options)

    def dropdown_handler(change):
        index = int(change.new)
        show_image(index)      
    
    def show_image(index):        
        left_image, right_image  = imread(left_image_paths[index]), imread(right_image_paths[index])
        
        try:
            show_image_napari(left_image, right_image)
        except:
            main_container.close()
            show_image_ipywidget(left_image, right_image) 
            
    def show_image_napari(image1, image2):
        viewer = napari.Viewer()
        viewer.add_image(image1)
        viewer.add_image(image2)            
            
    def plt_imshow(image):
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()
            
    def show_image_ipywidget(image1, image2):
        global main_container
        
        def update_left_image(change):
            left_checkbox.value = False
            index = int(change.new) if change else 0
            with output1:
                output1.clear_output(wait=True)
                plt_imshow(image1[index])

    
        def update_right_image(change):
            right_checkbox.value = False
            index = int(change.new) if change else 0
            with output2:
                output2.clear_output(wait=True)
                plt_imshow(image2[index])
                
        def right_checkbox_handler(change):
            if change.new == True:
                with output2:
                    output2.clear_output(wait=True)
                    img_display = image2.copy()
                    for ch in range(img_display.shape[0]):
                        img_display[ch,:,:] = (ch+1)*img_display[ch,:,:]
                    plt_imshow(img_display.max(axis=0))
            else:
                update_right_image(None)
                
        def left_checkbox_handler(change):
            if change.new == True:
                with output1:
                    output1.clear_output(wait=True)
                    plt_imshow(image1.max(axis=0))
            else:
                update_left_image(None)  
    
        output1 = widgets.Output()  
        output2 = widgets.Output()
    
        left_slider = set_IntSlider('ch:', 0, 0, len(image1) - 1, show=False)
        right_slider = set_IntSlider('ch:', 0, 0, len(image2) - 1, show=False)
        
        left_checkbox = set_checkbox(string_name = 'maximum intensity projection', default_value = False, show = False)
        right_checkbox = set_checkbox(string_name = 'maximum intensity projection', default_value = False, show = False)
    
        left_container = widgets.VBox(children=[output1, left_slider, left_checkbox])
        right_container = widgets.VBox(children=[output2, right_slider, right_checkbox])
        main_container = widgets.HBox(children=[left_container, right_container])
        
        
        update_left_image(None)
        update_right_image(None)
        display(main_container)            
        
        left_slider.observe(update_left_image, names='value')
        right_slider.observe(update_right_image, names='value')
        left_checkbox.observe(left_checkbox_handler, names='value')
        right_checkbox.observe(right_checkbox_handler, names='value')        
            
        
    dropdown_w.observe(dropdown_handler, names='value')    
    show_image(0)    
    

            

