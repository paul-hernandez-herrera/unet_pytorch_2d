import torch, warnings
import typing as t
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from ..models.UNet2d_model import Classic_UNet_2D
from ..loss.binary_loss import BinaryLoss
from torch.utils.data.dataset import Subset
from ..util import util
import numpy as np


def train_one_epoch(model, train_loader, optimizer, loss_functions, device, loss_reduction = 'sum'):
    #This is the main code responsible for updating the weights of the model for a single epoch
    
    model.train() #set the model in training mode
    epoch_loss = 0
    
    for imgs, targets in train_loader: 
        
        #we have a tensor in the train_loader, move to device
        imgs = imgs.to(device= device, dtype = torch.float32)
        targets = targets.to(device= device, dtype = torch.float32)
        
        optimizer.zero_grad()  # sets to zero the gradients of the optimizer
        
        # Forward pass
        network_output = model(imgs) 
        
        # Compute the loss
        if loss_reduction=='sum':
            loss = sum([f(network_output, targets) for f in loss_functions]) # compute the error between the network output and target output
        elif loss_reduction=='prod':
            losses = [loss_fn(network_output, targets) for loss_fn in loss_functions]
            loss = torch.prod(torch.stack(losses)) # compute the product of losses
        
        # Backward pass
        loss.backward() # compute the gradients given the loss value
        
        # update weights
        optimizer.step() # update the weights of models using the gradients and the given optimizer
        
        epoch_loss += loss.item()
    epoch_loss /= (len(loss_functions)*len(train_loader.dataset))
        
    return epoch_loss

def calculate_validation_loss(model, validation_loader, loss_functions, device, loss_reduction = 'sum'):
    model.eval() #set the model in evaluation mode
    val_loss = 0
    with torch.no_grad():  # disable gradient calculations during evaluation
        for imgs, targets in validation_loader: 
            
            #we have a tensor in the validation_loader, move to device
            imgs = imgs.to(device= device, dtype = torch.float32)
            targets = targets.to(device= device, dtype = torch.float32)
            
            network_output = model(imgs) #applying the model to the input images
            
            # Compute the loss
            if loss_reduction=='sum':
                loss = sum([f(network_output, targets) for f in loss_functions]) # compute the error between the network output and target output
            elif loss_reduction=='prod':
                losses = [loss_fn(network_output, targets) for loss_fn in loss_functions]
                loss = torch.prod(torch.stack(losses)) # compute the product of losses               
            
            val_loss += loss.item()
        val_loss /= len(validation_loader.dataset)
        
    return val_loss

def calculate_test_performance(model, test_loader, device, folder_output = None):
    model.eval() #set the model in evaluation mode
    
    if not(folder_output):
        folder_output = Path(folder_output,'test_results')
        folder_output.mkdir(parents=True, exist_ok=True)
    
    metric_dice = []
    binary_loss = BinaryLoss(option = 'dice', segmentation = True)
    img_id = 0
    with torch.no_grad():  # disable gradient calculations during evaluation
        for imgs, targets in test_loader:             
            #we have a tensor in the validation_loader, move to device
            imgs = imgs.to(device= device, dtype = torch.float32)
            targets = targets.to(device= device, dtype = torch.float32)
            
            network_output = model(imgs) #applying the model to the input images
            
            for j in range(0, targets.shape[0]):
                current_output = network_output[j,:,:,:][None]
                current_target = targets[j,:,:,:][None]
                
                val = 1-binary_loss(current_output, current_target).cpu().numpy()
                
                metric_dice.append(val)
                
                # Calculate probability map and convert to numpy array
                file_id = str(img_id).zfill(5)
                img_id+=1
                
                probability = torch.sigmoid(current_output).squeeze().cpu().numpy()
                util.imwrite(Path(folder_output, file_id + '_prob.tif'), (255 * probability).astype(np.uint8))
                util.imwrite(Path(folder_output, file_id + '_img.tif'), (255*imgs[j]).cpu().numpy().astype(np.uint8))
                util.imwrite(Path(folder_output, file_id + '_targets.tif'), (255 *targets[j]).cpu().numpy().astype(np.uint8))

    w_dataset = test_loader.dataset.dataset    
    index = np.array(test_loader.dataset.indices)        
    
    print("Printing test set ")
    for k, i in enumerate(index):
        print(f'test img_id: {k} --- file_name: {w_dataset.file_names[i]}')
        
    metric_dice = np.array(metric_dice)
    print(f"n_test = {len(test_loader)} ---- mean_error = {np.mean(metric_dice)} ---- std = {np.std(metric_dice)}" )
    print(metric_dice)

def get_model_outputdir(model_output_folder):
    if not model_output_folder:
        model_output_folder = Path(Path(__file__).absolute().parent, 'model_training_results', datetime.now().strftime('%Y_%m_%d_Time_%H_%M_%S'))
        model_output_folder.mkdir(parents=True, exist_ok=True)
        if Path(__file__).parent.stem != 'core_code':
            warnings.warn(f"We assume that the parent folder of function {Path(__file__).stem} is: core_code")
    return model_output_folder

def load_model(model_path, device = 'cpu', model_type = 'unet_2d'):
    state_dict = torch.load(model_path, map_location= device)

    n_channels_input = state_dict[list(state_dict.keys())[0]].size(1)
    n_channels_target = state_dict[list(state_dict.keys())[-1]].size(0)
    
    model = get_model(model_type, n_channels_input, n_channels_target).to(device= device)

    model.load_state_dict(state_dict)
    
    return model

# get automatic batch size --- implementation from
# https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1
#☺made small modifications
def get_batch_size(
    device: torch.device,
    input_shape: t.Tuple[int, int, int],
    output_shape: t.Tuple[int],
    dataset_size: int,
    model_type: str = 'resnet50',
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    
    model = get_model(model_type, input_shape[0], output_shape[0]).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 1
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
      
    return batch_size

def get_model(model_type, n_channels_input, n_channels_target):
    if model_type == 'unet_2d':
        model = Classic_UNet_2D(n_channels_input, n_channels_target)
        return model
    
def get_dataloader_file_names(dataset_loader, fullpath = True):
    #index and file_names from test images
    index = np.array(dataset_loader.dataset.indices)
    file_names = [dataset_loader.dataset.dataset.file_names[i] if fullpath else dataset_loader.dataset.dataset.file_names[i].stem for i in index]
    
    return file_names
    
    
    