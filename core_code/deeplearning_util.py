import torch
from . import util
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime

def train_one_epoch(model, train_loader, optimizer, list_loss_functions, device):
    #This is the main code responsible for updating the weights of the model for a single epoch
    
    model.train(True) #set the model in training mode
    epoch_loss = 0
    for batch_iter in train_loader: 
        imgs, targets = batch_iter #getting imgs and target output for current batch
        
        #we have a tensor in the train_loader, move to device
        imgs = imgs.to(device= device, dtype = torch.float32)
        targets = targets.to(device= device, dtype = torch.float32)
        #targets = targets.to(device=self.device, dtype=torch.long)
        
        optimizer.zero_grad()  #Sets the gradients of all optimized torch.Tensor s to zero.
        
        network_output = model(imgs) #applying the model to the input images
        
        loss = 0
        for i in range(0,len(list_loss_functions)):
            loss += list_loss_functions[i](network_output, targets) # compute the error between the network output and target output
        
        
        loss.backward() # compute the gradients given the loss value
        
        optimizer.step() # update the weights of models using the gradients and the given optimizer
        
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader.dataset) 
        
    return epoch_loss

def calculate_validation_loss(model, validation_loader, list_loss_functions, device):
    model.train(True) #set the model in training mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculations during evaluation
        for batch_iter in validation_loader: 
            imgs, targets = batch_iter #getting imgs and target output for current batch
            
            #we have a tensor in the validation_loader, move to device
            imgs = imgs.to(device= device, dtype = torch.float32)
            targets = targets.to(device= device, dtype = torch.float32)
            
            network_output = model(imgs) #applying the model to the input images
            
            loss = 0
            for i in range(0,len(list_loss_functions)):
                loss += list_loss_functions[i](network_output, targets) # compute the error between the network output and target output
            
            val_loss += loss.item()
        val_loss /= len(validation_loader.dataset)
        
    return val_loss

def predict_model(model, input_path, folder_output=None, device = 'cpu'):
    file_paths = util.get_image_file_paths(input_path)
    
    # Set output folder path
    folder_output = folder_output or Path(file_paths[0]).parent / 'output'
    folder_output.mkdir(parents=True, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient calculations during evaluation
    with torch.no_grad():  
        for img_file_path in file_paths: 
            # Load input image as tensor
            img = torch.tensor(util.imread(img_file_path).astype(np.float32)).unsqueeze(0).to(device=device)
            
            # Apply model to input image
            network_output = model(img) 
            
            probability = torch.sigmoid(network_output).squeeze().cpu().numpy()
            
            # Save output probability map as image
            output_file_path = Path(folder_output, Path(img_file_path).stem + '_prob.tif')
            
            util.imwrite(output_file_path, 255 * probability)
            
            print(output_file_path)
    return 

def get_model_outputdir(model_output_folder):
    if model_output_folder is None:
        folder_root = Path(__file__).absolute().parent
        current_time = datetime.now().strftime('%Y_%m_%d_Time_%H_%M_%S')
        model_output_folder = Path(folder_root, 'model_training_results', current_time)
        model_output_folder.mkdir(parents=True, exist_ok=True)
        if Path(__file__).parent.stem != 'core_code':
            warnings.warn(f"We assume that the parent folder of function {Path(__file__).stem} is: core_code")
    return model_output_folder