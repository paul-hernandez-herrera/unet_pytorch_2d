import torch, warnings
from pathlib import Path
from datetime import datetime

from ..parameters_interface import ipwidget_basic
from ..parameters_interface.parameters_widget import parameters_device
from ..UNet_2D.UNet2d_model import Classic_U_Net_2D
from ..predict import predict_model


def train_one_epoch(model, train_loader, optimizer, loss_functions, device):
    #This is the main code responsible for updating the weights of the model for a single epoch
    
    model.train() #set the model in training mode
    epoch_loss = 0
    
    for batch in train_loader: 
        imgs, targets = batch #getting imgs and target output for current batch
        
        #we have a tensor in the train_loader, move to device
        imgs = imgs.to(device= device, dtype = torch.float32)
        targets = targets.to(device= device, dtype = torch.float32)
        #targets = targets.to(device=self.device, dtype=torch.long)
        
        optimizer.zero_grad()  # sets to zero the gradients of the optimizer
        
        # Forward pass
        network_output = model(imgs) 
        
        # Compute the loss
        loss = sum([f(network_output, targets) for f in loss_functions]) # compute the error between the network output and target output
        
        # Backward pass
        loss.backward() # compute the gradients given the loss value
        
        # update weights
        optimizer.step() # update the weights of models using the gradients and the given optimizer
        
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader.dataset)
        
    return epoch_loss

def calculate_validation_loss(model, validation_loader, loss_functions, device):
    model.eval() #set the model in evaluation mode
    val_loss = 0
    with torch.no_grad():  # disable gradient calculations during evaluation
        for batch in validation_loader: 
            imgs, targets = batch #getting imgs and target output for current batch
            
            #we have a tensor in the validation_loader, move to device
            imgs = imgs.to(device= device, dtype = torch.float32)
            targets = targets.to(device= device, dtype = torch.float32)
            
            network_output = model(imgs) #applying the model to the input images
            
            loss = sum([loss_fn(network_output, targets) for loss_fn in loss_functions]) # compute the error between the network output and target output
            
            val_loss += loss.item()
        val_loss /= len(validation_loader.dataset)
        
    return val_loss

def get_model_outputdir(model_output_folder):
    if not model_output_folder:
        model_output_folder = Path(Path(__file__).absolute().parent, 'model_training_results', datetime.now().strftime('%Y_%m_%d_Time_%H_%M_%S'))
        model_output_folder.mkdir(parents=True, exist_ok=True)
        if Path(__file__).parent.stem != 'core_code':
            warnings.warn(f"We assume that the parent folder of function {Path(__file__).stem} is: core_code")
    return model_output_folder

def load_model(model_path, device = 'cpu'):
    state_dict = torch.load(model_path, map_location= device)

    n_channels_input = state_dict[list(state_dict.keys())[0]].size(1)
    n_channels_target = state_dict[list(state_dict.keys())[-1]].size(0)
    
    model = Classic_U_Net_2D(n_channels_input, n_channels_target).to(device= device)
    
    return model
    
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
        
        model = load_model(model_path = model_path, device = device)
    
        # Predict images and return list of output file paths
        file_paths = predict_model(model, self.folder_path_w.value, folder_output = self.folder_output_w.value, device = device)
        
        return file_paths
    
    