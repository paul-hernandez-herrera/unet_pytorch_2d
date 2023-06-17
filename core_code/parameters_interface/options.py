from torch.optim import SGD, Adam, NAdam, RMSprop
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR, StepLR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from ..loss.binary_loss import BinaryLoss
from ..datasets.data_augmentation_segmentation import augmentation_segmentation_task
from ..datasets.Dataset import CustomImageDataset
from typing import Dict
import copy



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

def get_lr_scheduler(option_name: str, optimizer, **kwargs) -> object:
    default_params: Dict[str, object] = {
        'reduce_on_plateau': {'mode': 'min', 'factor': 0.1, 'patience': 10},
        'cyclic': {'base_lr': 0.0001, 'max_lr': 0.01, 'step_size_up': 50, 'cycle_momentum': False},
        'cosine_annealing': {'T_max': 50},
        'step': {'step_size': 10, 'gamma': 0.1}
    }
    params_optimizer = default_params[option_name]

    #update to new values
    params = {k: kwargs[k] if k in kwargs else v for k, v in params_optimizer.items()}

    
    if option_name == 'reduce_on_plateau':
        return ReduceLROnPlateau(optimizer, **params)
    elif option_name == 'cyclic':
        return CyclicLR(optimizer, **params)
    elif option_name == 'cosine_annealing':
        return CosineAnnealingLR(optimizer, **params)
    elif option_name == 'step':
        return StepLR(optimizer, **params)
    else:
        raise ValueError(f"Invalid option name: {option_name}")

###################################################################

def get_loss_function(option_name: str):
    
    if option_name == 'cross_entropy': 
        loss_function = [CrossEntropyLoss()]
    elif option_name == 'dice_loss': 
        loss_function = [BinaryLoss('dice')]
    elif option_name == 'dice_cross': 
        loss_function = [BinaryLoss('dice'), CrossEntropyLoss()]
    elif option_name == 'BCEWithLogitsLoss': 
        loss_function = [BCEWithLogitsLoss()]
    elif option_name == 'dice_BCE': 
        loss_function = [BinaryLoss('dice'), BCEWithLogitsLoss()]    
    else:
        raise Exception("Sorry, " + option_name + " not recognized as loss function option.")
        
    return loss_function

###################################################################

def get_data_augmentation(enable_data_augmentation = True, **kargs): 
    if enable_data_augmentation:
        return augmentation_segmentation_task(**kargs)
    return None

###################################################################

def get_split_training_val_test_sets(train_dataset, val_par, test_par):
    train_dataset = copy.deepcopy(train_dataset)
    train_dataset, val_set = split_dataset(train_dataset, val_par, test_par)
    train_dataset, test_set = split_dataset(train_dataset, test_par, val_par)    
    #validation and training set to be use without dataaugmentation
    if (val_par["type"]=='percentage_training_set') and (test_par["type"]=='percentage_training_set'):
        perc1 = val_par["per_val"]
        perc2 = test_par["per_val"]
        train_dataset, val_set, test_set = random_split(train_dataset, [1-perc1-perc2, perc1, perc2])
        
    train_dataset = copy.deepcopy(train_dataset)
    if val_set: 
        val_set.dataset.set_data_augmentation(augmentation_flag = False)
    if isinstance(val_set, Subset):
        val_set.dataset.set_data_augmentation(augmentation_flag = False)
    return train_dataset, val_set, test_set

def split_dataset(train_dataset, parameters_1, parameters_2):
    if parameters_1["type"] == 'None':
        return train_dataset, None
    elif parameters_1["type"] == 'folder_path':
        return train_dataset, CustomImageDataset(parameters_1["folder_input"], parameters_1["folder_target"])
    elif parameters_1["type"]=='percentage_training_set' and parameters_2["type"]=='None':
        perc1 = parameters_1["per_val"]
        return random_split(train_dataset, [1-perc1, perc1])
    return train_dataset, None