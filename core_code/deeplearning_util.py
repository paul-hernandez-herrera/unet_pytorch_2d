import torch
from . import util
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime

def train_one_epoch(model, train_dataloader, optimizer, list_loss_functions, device):
    #This is the main code responsible for updating the weights of the model for a single epoch
    
    model.train(True) #set the model in training mode
    epoch_loss = 0
    for batch_iter in train_dataloader: 
        imgs, targets = batch_iter #getting imgs and target output for current batch
        
        #we have a tensor in the train_dataloader, move to device
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
        
    return epoch_loss

def calculate_validation_loss(model, validation_dataloader, list_loss_functions, device):
    model.train(True) #set the model in training mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculations during evaluation
        for batch_iter in validation_dataloader: 
            imgs, targets = batch_iter #getting imgs and target output for current batch
            
            #we have a tensor in the train_dataloader, move to device
            imgs = imgs.to(device= device, dtype = torch.float32)
            targets = targets.to(device= device, dtype = torch.float32)
            
            network_output = model(imgs) #applying the model to the input images
            
            loss = 0
            for i in range(0,len(list_loss_functions)):
                loss += list_loss_functions[i](network_output, targets) # compute the error between the network output and target output
            
            val_loss += loss.item()
        
    return val_loss

def predict_model(model, input_path, folder_output, device = 'cpu'):
    file_paths = util.get_image_file_paths(input_path)
    
    model.train(False) #set the model in training mode
    with torch.no_grad():  # Disable gradient calculations during evaluation
        for img_file_name in file_paths: 
            
            img = torch.tensor(util.imread(img_file_name).astype(np.float32)).float()
            
            img = img[None]
            
            img = img.to(device=device, dtype=torch.float32)
            
            network_output = model(img) #applying the model to the input images
            
            probability = torch.sigmoid(network_output).squeeze().numpy()
            
            print(Path(folder_output, Path(img_file_name).stem + '_prob.tif'))
            util.imwrite(Path(folder_output, Path(img_file_name).stem + '_prob.tif'), 255*probability)
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