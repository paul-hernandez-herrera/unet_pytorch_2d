from .util import util
import numpy as np
from pathlib import Path
import torch
from .util.deeplearning_util import load_model
from .parameters_interface import ipwidget_basic
from .parameters_interface.parameters_widget import parameters_device
from .util.preprocess import preprocess_image

def predict_model(input_path, model= None, model_path = None, output_folder=None, device = 'cpu', enable_preprocess = False):
    file_paths = util.get_image_file_paths(input_path)
    
    # Set output folder path
    output_folder = output_folder or Path(file_paths[0]).parent / 'output'
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    if not model:
        model = load_model(model_path = model_path, device = device)
    
    # Disable gradient calculations during evaluation
    output_file_paths = []
    with torch.no_grad():  
        for img_file_path in file_paths:
            
            # reading image
            input_img = util.imread(img_file_path).astype(np.float32)            
            input_img = input_img[None] if input_img.ndim ==  2 else input_img
            
            #preprocess image if required
            if enable_preprocess:
                input_img = preprocess_image(input_img, percentile_range = [1,99])              
            
            # Load input image as tensor
            img = torch.tensor(input_img).unsqueeze(0).to(device=device)
            
            # Apply model to input image
            network_output = model(img) 
            
            # Calculate probability map and convert to numpy array
            probability = torch.sigmoid(network_output).squeeze().cpu().numpy()
            
            # Save output probability map as image
            output_file_path = Path(output_folder, Path(img_file_path).stem + '_prob.tif')
            util.imwrite(output_file_path, 255 * probability)
            
            print(output_file_path)
            output_file_paths.append(output_file_path)
    return {"inputs": file_paths, "outputs": output_file_paths}


class PredictSegmentationInteractive:
    def __init__(self):
        #setting the parameters required to predict images
        # Set parameters required to predict images
        self.model_path_w = ipwidget_basic.set_text('Model path:', 'Insert path here')
        self.folder_path_w = ipwidget_basic.set_text('Input path:', 'Insert path here')
        self.folder_output_w = ipwidget_basic.set_text('Output path:', 'Insert path here')
        self.device_w = parameters_device()
        
    def run(self):
        model_path = self.model_path_w.value
        device = self.device_w.get_device()
    
        # Predict images and return list of output file paths
        file_paths = predict_model(self.folder_path_w.value, 
                                   model_path = model_path, 
                                   output_folder = self.folder_output_w.value, 
                                   device = device,
                                   enable_preprocess = False)
        
        return file_paths

if __name__== '__main__':
    
    #to define
    print('to define command line options')