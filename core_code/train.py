import argparse 
import torch
from pathlib import Path
import copy
from .deeplearning_util import train_one_epoch, compute_validation_loss, get_model_outputdir
import platform
import torchvision
from torch.utils.tensorboard import SummaryWriter

def train_model(model, 
                train_dataloader, 
                list_loss_functions,
                optimizer,
                output_folder = None,
                device = None,
                epochs = 100, 
                validation_dataloader = [],                  
                model_checkpoint = False, 
                model_checkpoint_frequency = 10,
                lr_scheduler = None):
    
    output_folder = get_model_outputdir(get_model_outputdir)
    
    writer = SummaryWriter()
    
    if (torch.__version__>= '2.0.0') & (platform.system() != 'Windows'):
        model = torch.compile(model, mode= 'reduce-overhead')
    
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    current_minimum_loss = float('inf')
    
    for epoch in range(1, epochs+1):
        print('Running iteration ' + str(epoch) + '/' +str(epochs))
        model_loss = train_one_epoch(model, train_dataloader, optimizer, list_loss_functions, device)
        print(f"epoch loss: {model_loss}")
        
        if len(validation_dataloader)>0:
            # if there is data available in the validation dataloader, the validation loss should be computed. Otherwise we use the loss value from the test data
            model_loss = compute_validation_loss(model, validation_dataloader, list_loss_functions, device)
        
        if lr_scheduler != None:
            lr_scheduler.step(model_loss)
            
        if model_loss < current_minimum_loss:
            best_model_state_dict, current_minimum_loss, best_epoch = copy.deepcopy(model.state_dict()), model_loss, epoch
            
        if model_checkpoint & ((epoch%model_checkpoint_frequency) ==0):
            torch.save(model.state_dict(), Path(output_folder, f"model_{epoch}.pth"))
        
        print(optimizer.param_groups[0]['lr'])
            
    torch.save(model.state_dict(), Path(output_folder, f"last_model_e{epochs}.pth"))
    torch.save(best_model_state_dict, Path(output_folder, f"best_model_e{best_epoch}.pth"))
    return model             
        

    
    
    
if __name__== '__main__':
    
    #to define
    parser = argparse.ArgumentParser(description='Main code to train the U-Net given a training set of 2D-input images and target image.')
    parser.add_argument('--folder_imgs', default=[], type=str, help='Path to the folder containing the input-images in the training set', required = True)
    parser.add_argument('--folder_target', default=[], type=str, help='Path to the folder containing the target-images in the training set.', required = True)
    args = parser.parse_args()