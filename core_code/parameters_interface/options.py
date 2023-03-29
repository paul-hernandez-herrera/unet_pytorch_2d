from torch.optim import SGD, Adam, NAdam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR, StepLR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from ..loss.dice_loss import DiceLoss2D
from ..UNet_2D.data_augmentation import augmentation_segmentation_task

def get_optimizer(option_name: str, 
                  model,
                  lr = 0.0001,
                  weight_decay = 1e-8,
                  momentum = 0.9,
                  betas = (0.9, 0.999)):
    
    if option_name == 'SGD':
        optimizer = SGD(model.parameters(), lr = lr, momentum = momentum)
    elif option_name == 'Adam':
        optimizer = Adam(model.parameters(), lr = lr, betas = betas)
    elif option_name == 'Nesterov_Adam':
        optimizer = NAdam(model.parameters(), lr = lr, betas = betas)
    elif option_name == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
    else:
        raise Exception("Sorry, " + option_name + " not recognized as optimizer option.")
    return optimizer

###################################################################

def get_lr_scheduler(option_name: str,
                     optimizer, 
                     factor = 0.1,
                     patience = 10,
                     base_lr = 0.0001,
                     max_lr = 0.01,
                     step_size_up = 50,
                     T_max = 50,
                     step_size = 10,
                     gamma = 0.1,
                     mode = 'min'):
    if option_name == 'reduce_on_plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode = mode, factor = factor, patience = patience)
    elif option_name == 'cyclic':
        lr_scheduler = CyclicLR(optimizer, mode = mode, base_lr = base_lr, max_lr = max_lr, step_size_up = step_size_up)
    elif option_name == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max = T_max)
    elif option_name == 'step':
        lr_scheduler = StepLR(optimizer, step_size = step_size, gamma = gamma)
    else:
        raise Exception("Sorry, " + option_name + " not recognized as lr scheduler option.")
    return lr_scheduler

###################################################################

def get_loss_function(option_name: str):
    
    if option_name == 'cross_entropy': 
        loss_function = [CrossEntropyLoss()]
    elif option_name == 'dice_loss': 
        loss_function = [DiceLoss2D()]
    elif option_name == 'dice_cross': 
        loss_function = [DiceLoss2D(), CrossEntropyLoss()]
    elif option_name == 'BCEWithLogitsLoss': 
        loss_function = [BCEWithLogitsLoss()]
    elif option_name == 'dice_BCE': 
        loss_function = [DiceLoss2D(), BCEWithLogitsLoss()]    
    else:
        raise Exception("Sorry, " + option_name + " not recognized as loss function option.")
        
    return loss_function

###################################################################

def get_data_augmentation(hflip_flag = True,
                          vflip_flag = True,
                          shear_flag = True,
                          zoom_flag = True,
                          shear_angle = [-5, 5],
                          zoom_range = [0.8, 1.2],
                          data_augmentation_flag = True): 
    
    data_augmentation_object = augmentation_segmentation_task()
            
    if data_augmentation_flag:
        # Setting the data augmentation flags
        data_augmentation_object.hflip_flag = hflip_flag
        data_augmentation_object.vflip_flag = vflip_flag
        data_augmentation_object.shear_flag = shear_flag
        data_augmentation_object.zoom_flag = zoom_flag
        
        data_augmentation_object.shear_angle = shear_angle
        data_augmentation_object.zoom_range = zoom_range
    else:
        data_augmentation_object = None
        
    return data_augmentation_object

###################################################################