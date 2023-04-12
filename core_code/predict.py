from .util import util
import numpy as np
from pathlib import Path
import torch
from .util.deeplearning_util import load_model

def predict_model(input_path, model= None, model_path = None, output_folder=None, device = 'cpu'):
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
            # Load input image as tensor
            img = torch.tensor(util.imread(img_file_path).astype(np.float32)).unsqueeze(0).to(device=device)
            
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


if __name__== '__main__':
    
    #to define
    print('to define command line options')