import torch, warnings
import typing as t
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from ..models.UNet2d_model import Classic_UNet_2D



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
    
    model = Classic_UNet_2D(n_channels_input, n_channels_target).to(device= device)

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
    
    model = get_model(model_type, input_shape[0], output_shape[0])
    model.to(device)
    model.train(True)
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
    
    
    