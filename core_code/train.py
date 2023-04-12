import argparse, torch, copy, platform
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from .util.deeplearning_util import train_one_epoch, calculate_validation_loss, get_model_outputdir



def train_model(model, 
                train_loader, 
                loss_functions,
                optimizer,
                output_dir = None,
                device = None,
                epochs = 100, 
                validation_loader = None,                  
                save_checkpoint = False, 
                checkpoint_frequency = 10,
                lr_scheduler = None):
    
    output_dir = get_model_outputdir(output_dir)
    
    writer = SummaryWriter(log_dir = Path(output_dir, 'tensorboard/'))
    
    if (torch.__version__>= '2.0.0') & (platform.system() != 'Windows'):
        model = torch.compile(model, mode= 'reduce-overhead')
    
    device = device or "cuda" if torch.cuda.is_available() else "cpu"
        
    current_minimum_loss = float('inf')
    
    for epoch in range(1, epochs+1):
        print(f"Running iteration {epoch}/{epochs}")
        model_loss = train_one_epoch(model, train_loader, optimizer, loss_functions, device)
        print(f"epoch loss: {model_loss} --- lr = {optimizer.param_groups[0]['lr']}")
        
        val_loss = calculate_validation_loss(model, validation_loader, loss_functions, device) if validation_loader else None
        val_loss = val_loss or model_loss
        
        if lr_scheduler:
            lr_scheduler.step(val_loss)
            
        if val_loss < current_minimum_loss:
            best_model_state_dict, current_minimum_loss, best_epoch = copy.deepcopy(model.state_dict()), val_loss, epoch
            
        if save_checkpoint & ((epoch%checkpoint_frequency) ==0):
            torch.save(model.state_dict(), Path(output_dir, f"model_{epoch}.pth"))
        
        writer.add_scalars('training model', {'train loss': model_loss, 'val loss': val_loss}, epoch)
        
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'] , epoch)
            
    writer.close()
    torch.save(model.state_dict(), Path(output_dir, f"last_model_e{epochs}.pth"))
    torch.save(best_model_state_dict, Path(output_dir, f"best_model_e{best_epoch}.pth"))
    return model             
        

    
    
    
if __name__== '__main__':
    
    #to define
    parser = argparse.ArgumentParser(description='Main code to train the U-Net given a training set of 2D-input images and target image.')
    parser.add_argument('--folder_imgs', default=[], type=str, help='Path to the folder containing the input-images in the training set', required = True)
    parser.add_argument('--folder_target', default=[], type=str, help='Path to the folder containing the target-images in the training set.', required = True)
    args = parser.parse_args()