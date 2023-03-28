import ipywidgets as widgets
import torch
from IPython.display import display
from ..loss import dice_loss
from ..UNet_2D.data_augmentation import augmentation_segmentation_task
from ..UNet_2D.Dataset import CustomImageDataset
from . import ipwidget_basic

    
################################################################################

class parameters_training_images():
    def __init__(self):
        print('------------------------------')
        print('\033[47m' '\033[1m' 'REQUIRED PARAMETERS' '\033[0m')
        print('------------------------------')
        self.folder_input_w = ipwidget_basic.set_text('Folder path input images:', 'Insert path here')   
        self.folder_target_w = ipwidget_basic.set_text('Folder path target mask:', 'Insert path here')
    
    def get(self):
        p = {"folder_input"     : self.folder_input_w.value,
             "folder_target"    : self.folder_target_w.value}
        return p

################################################################################

class parameters_create_training_set():
    def __init__(self):    
        print('------------------------------')
        print('\033[47m' '\033[1m' 'REQUIRED PARAMETERS' '\033[0m')
        print('------------------------------')
        self.folder_brightfield_w    = ipwidget_basic.set_text('Folder images - brightfield: ', 'Insert path here')
        self.folder_fluorescence_w   = ipwidget_basic.set_text('Folder images - fluorescence: ', 'Insert path here')
        self.folder_head_w           = ipwidget_basic.set_text('Folder target images - head: ', 'Insert path here')
        self.folder_flagellum_w      = ipwidget_basic.set_text('Folder target images - flagellum: ', 'Insert path here')
        self.folder_output_w         = ipwidget_basic.set_text('Folder output: ', 'Insert path here')
    
    def get(self):
        p = {"folder_brightfield"    : self.folder_brightfield_w.value,
             "folder_fluorescence"   : self.folder_fluorescence_w.value,
             "folder_head"           : self.folder_head_w.value,
             "folder_flagellum"      : self.folder_flagellum_w.value,
             "folder_output"         : self.folder_output_w.value}
        return p

################################################################################.


class parameters_model_training():
    def __init__(self, model, training_dataset, n_channels_target):
        print('------------------------------')
        print('\033[47m' '\033[1m' 'REQUIRED PARAMETERS' '\033[0m')
        print('------------------------------')
        
        self.model_saving = parameters_model_saving()
        
        print('------------------------------')
        print('\033[47m' '\033[1m' 'OPTIONAL PARAMETERS' '\033[0m')
        print('------------------------------')        
    
        self.batch_w = ipwidget_basic.set_Int('Batch size: ', 8)
        self.number_epochs_w = ipwidget_basic.set_Int('Number of epochs: ', 100) 
        
        print('------------------------------')
        ## setting validation set
        self.validation = parameters_validation_set(training_dataset)
        print('------------------------------')
        
        
        self.criterion_loss = parameters_loss_function(n_channels_target)
    
        ## setting optimizer
        print('------------------------------')
        self.optimizer = parameters_optimizer(model)
        print('------------------------------')
        
        ## lr scheduler
        self.lr_scheduler = parameters_lr_scheduler()
        

    
    def get(self, str_id):
        if str_id == 'batch': return self.batch_w.value
        if str_id == 'epochs': return self.number_epochs_w.value
        if str_id == 'loss_function': return self.criterion_loss.get()["loss_function"]
        if str_id == 'optimizer': return self.optimizer.get()
        if str_id == 'train_dataset': return self.validation.get()["train_dataset"]
        if str_id == 'validation_dataset': return self.validation.get()["validation_dataset"]
        if str_id == 'model_output_folder': return self.model_saving.get("model_output_folder")         
        if str_id == 'model_checkpoint': return self.model_saving.get("model_checkpoint")
        if str_id == 'model_checkpoint_frequency': return self.model_saving.get("model_checkpoint_frequency")

################################################################################

class parameters_device():
    def __init__(self):
        ##SETTING DEVICE
        device_options = [('CPU', 'cpu')]    
        #checking if torch is available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count(),0,-1):
                device_options.insert(0, (torch.cuda.get_device_name(i-1), 'cuda:'+str(i-1)))
        
        #setting the option to get the device
        self.device_w = ipwidget_basic.set_dropdown('Device: ', device_options)
    
    def get_device(self):
        return self.device_w.value
    
################################################################################

class parameters_optimizer():
    def __init__(self, model):
        self.model = model
        
        optimizer_options = [('Adam', 'Adam'),
                             ('Nesterov_Adam', 'Nesterov_Adam'),
                             ('Stochastic Gradient Descent', 'SGD'),
                             ('RMSprop', 'RMSprop')]
        
        self.optimizer_w = ipwidget_basic.set_dropdown('Optimizer: ', optimizer_options)
        self.optimizer_w.observe(self.dropdown_handler_optimizer, names='value')

        #optional parameters for the Optimizer SGD
        self.learning_rate_w = ipwidget_basic.set_Float_Bounded('Learning rate: ', 0.0001, 0, 1, 0.001)
        
        #optional parameters for the Optimizer Adam
        self.beta1_w = ipwidget_basic.set_Float_Bounded('Beta 1: ', 0.9, 0, 1, 0.01)
        self.beta2_w = ipwidget_basic.set_Float_Bounded('Beta 2: ', 0.999, 0, 1, 0.0001)
        
        
        #optional parameters for the Optimizer RMSprop
        self.momentum_w = ipwidget_basic.set_Float_Bounded('Momentum: ', 0.9, 0, 1, 0.01)
        self.weigth_decay_w = ipwidget_basic.set_Float_Bounded('weight decay: ', 1e-8, 0, 1, 1e-8)
        
       
        #container for the parameters. Default is Adam
        self.main_container = widgets.HBox(children= [self.learning_rate_w, self.beta1_w, self.beta2_w])
        
        display(self.main_container)
        
    def dropdown_handler_optimizer(self, change):
        if change.new == 'SGD':
            self.main_container.children = [self.learning_rate_w, self.momentum_w]
        elif change.new == 'Adam':
            self.main_container.children = [self.learning_rate_w, self.beta1_w, self.beta2_w]
        elif change.new == 'Nesterov_Adam':            
            self.main_container.children = [self.learning_rate_w, self.beta1_w, self.beta2_w]
        elif change.new == 'RMSprop':
            self.main_container.children = [self.learning_rate_w, self.weigth_decay_w, self.momentum_w]     
    
    def get(self):
        if self.optimizer_w.value == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr= self.learning_rate_w.value, momentum = self.momentum_w.value)
        elif self.optimizer_w.value == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr= self.learning_rate_w.value, betas=(self.beta1_w.value, self.beta2_w.value))
        elif self.optimizer_w.value == 'Nesterov_Adam':
            optimizer = torch.optim.NAdam(self.model.parameters(), lr= self.learning_rate_w.value, betas=(self.beta1_w.value, self.beta2_w.value))
        elif self.optimizer_w.value == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr= self.learning_rate_w.value, weight_decay= self.weigth_decay_w.value, momentum = self.momentum_w.value)
        return optimizer

################################################################################

class parameters_loss_function():
    def __init__(self, n_channels_target):
        
        # options for loss functions depending on the problem to solve
        self.multiclass_options = [('Cross entropy loss', 'cross_entropy'),
                                   ('Dice loss', 'dice_loss'),
                                   ('Dice + Cross Entropy', 'dice_cross')]
        
        self.singleClass_options = [('Binary cross entropy (BCE)', 'BCEWithLogitsLoss'),
                                    ('Dice loss', 'dice_loss'),
                                    ('Dice + BCE', 'dice_BCE')]        
        
        if n_channels_target ==1:
            loss_function_options = self.singleClass_options
        else:
            loss_function_options = self.multiclass_options
    
        self.loss_w = ipwidget_basic.set_dropdown('Loss function: ', loss_function_options)
        
        
    def get(self):
        if self.loss_w.value == 'cross_entropy': 
            loss_function = [torch.nn.CrossEntropyLoss()]
        elif self.loss_w.value == 'dice_loss': 
            loss_function = [dice_loss.DiceLoss2D()]
        elif self.loss_w.value == 'dice_cross': 
            loss_function = [dice_loss.DiceLoss2D(), torch.nn.CrossEntropyLoss()]
        elif self.loss_w.value == 'BCEWithLogitsLoss': 
            loss_function = [torch.nn.BCEWithLogitsLoss()]
        elif self.loss_w.value == 'dice_BCE': 
            loss_function = [dice_loss.DiceLoss2D(), torch.nn.BCEWithLogitsLoss()]
            
        p = {"loss_function"    : loss_function}
        return p

################################################################################

class parameters_validation_set():
    def __init__(self, train_dataset):
        
        self.train_dataset = train_dataset
        
        # options for loss functions
        validation_options = [('None', 'None'),
                              ('Folder paths', 'folder_path'),
                              ('% of training set', 'percentage_training_set')
                              ]
        
        self.validation_w = ipwidget_basic.set_dropdown('Validation: ', validation_options)
        
        self.folder_input_w  = ipwidget_basic.set_text('Folder images: ', 'Insert path here', show = False)
        self.folder_target_w = ipwidget_basic.set_text('Folder target: ', 'Insert path here', show = False)
        
        self.perc_training_set = ipwidget_basic.set_Float_Bounded(' ', 0.05, 0, 1, 0.01)
        
        self.main_container = widgets.VBox(children= [])
        
        self.validation_w.observe(self.dropdown_handler_validation, names='value')
        display(self.main_container)
        
        
    def dropdown_handler_validation(self, change):
        if change.new == 'folder_path':
            self.main_container.children = [self.folder_input_w, self.folder_target_w]
        elif change.new == 'percentage_training_set':
            self.main_container.children = [self.perc_training_set]
        elif change.new == 'None':
            self.main_container.children = []
            
    def split_training_validation(self):
        if self.validation_w.value == 'folder_path':
            new_train_dataset = self.train_dataset
            validation_dataset = CustomImageDataset(self.folder_input_w.value, self.folder_target_w)
        elif self.validation_w.value == 'percentage_training_set':
            per_val = self.perc_training_set.value
            #to generate the same validation_set
            generator_seed = torch.Generator().manual_seed(1)
            new_train_dataset, validation_dataset = torch.utils.data.random_split(self.train_dataset, [1-per_val, per_val], generator = generator_seed)
        elif self.validation_w.value == 'None':
            new_train_dataset = self.train_dataset
            validation_dataset = None            
        
        return new_train_dataset, validation_dataset
            
    def get(self):
        
        new_train_dataset, validation_dataset = self.split_training_validation()
            
        p = {"train_dataset"    : new_train_dataset,
             "validation_dataset"  : validation_dataset}
        return p                
            


################################################################################
          
class parameters_data_augmentation():
    def __init__(self):
        
        #to retrieve default values
        dummy = augmentation_segmentation_task()
        
        # options for loss functions
        self.data_augmentation_w = ipwidget_basic.set_checkbox('Data augmentation', False, show = False)
        
        self.hflip_flag_w = ipwidget_basic.set_checkbox('Horizontal flip', True, show = False)
        self.vflip_flag_w = ipwidget_basic.set_checkbox('Vertical flip', True, show = False)
        self.shear_flag_w = ipwidget_basic.set_checkbox('Shear', True, show = False)
        self.shear_angle_w = ipwidget_basic.set_intSlider('Angle', dummy.shear_angle[0], dummy.shear_angle[1], -180, 180, show = False)
        self.zoom_flag_w = ipwidget_basic.set_checkbox('Zoom', True, show = False)
        self.zoom_w = ipwidget_basic.set_FloatRangeSlider('Zoom', dummy.zoom_range[0], dummy.zoom_range[1], 0.1, 2, show = False)
        
        
        self.main_container = widgets.VBox(children= [self.data_augmentation_w])
        
        
        self.shear_container = widgets.HBox(children= [self.shear_flag_w, self.shear_angle_w])
        self.zoom_container = widgets.HBox(children= [self.zoom_flag_w, self.zoom_w])
        self.options_container = widgets.VBox(children= [self.hflip_flag_w, self.vflip_flag_w, self.shear_container, self.zoom_container])
        
        self.data_augmentation_w.observe(self.dropdown_handler_augmentation, names='value')
        display(self.main_container)
        
    def dropdown_handler_augmentation(self, change):
        if self.data_augmentation_w.value == True:
            self.main_container.children = [self.data_augmentation_w, self.options_container]
        else:
            self.main_container.children = [self.data_augmentation_w]
            
    def get(self):
        self.data_augmentation_w.value
        
        if self.data_augmentation_w.value == True:
            self.data_augmentation_object = augmentation_segmentation_task()
            
            # Setting the data augmentation flags
            self.data_augmentation_object.hflip_flag = self.hflip_flag_w.value
            self.data_augmentation_object.vflip_flag = self.vflip_flag_w.value
            self.data_augmentation_object.shear_flag = self.shear_flag_w.value
            self.data_augmentation_object.zoom_flag = self.zoom_flag_w.value
            
            self.data_augmentation_object.shear_angle = self.shear_angle_w.value
            self.data_augmentation_object.zoom_range = self.zoom_w.value
        else:
            self.data_augmentation_object = None
            
        p = {"data_augmentation_flag"     : self.data_augmentation_w.value,
             "data_augmentation_object"   : self.data_augmentation_object}
            
        return p
    
class parameters_model_saving():
    def __init__(self):
        self.model_output_w = ipwidget_basic.set_text('Model output folder : ', 'Insert path here', show = False)
        self.model_checkpoint_w = ipwidget_basic.set_checkbox('Model checkpoint interval. ', False, show = False)
        
        self.model_checkpoint_frequency = ipwidget_basic.set_Int('Frequency', 10, show  = False)
        
        self.epoch_container = widgets.HBox(children= [self.model_checkpoint_w])
        self.main_container = widgets.VBox(children= [self.model_output_w, self.epoch_container])
        
        self.model_checkpoint_w.observe(self.handler_model_saving, names='value') 
        display(self.main_container)
        
    def handler_model_saving(self, change):
        if change.new:
            self.epoch_container.children = [self.model_checkpoint_w, self.model_checkpoint_frequency]
        else:
            self.epoch_container.children = [self.model_checkpoint_w]
            
    def get(self, str_id):
        if str_id == 'model_output_folder': return self.model_output_w.value         
        if str_id == 'model_checkpoint': return self.model_checkpoint_w.value
        if str_id == 'model_checkpoint_frequency': return self.model_checkpoint_frequency.value
        
################################################################################        

class parameters_lr_scheduler():
    def __init__(self):
        
        lr_scheduler_options = [('reduce LR on plateau', 'reduce_on_plateau'),
                                ('Cyclic LR', 'cyclic'),
                                ('Cosine Annealing LR', 'cosine_annealing'),
                                ('Step LR', 'step')
                                ]
        
        self.lr_scheduler_w = ipwidget_basic.set_dropdown('lr schedulers: ', lr_scheduler_options)
        self.lr_scheduler_w.observe(self.dropdown_handler_lr_scheduler, names='value')
        

        # parameters plateau
        self.factor_w = ipwidget_basic.set_Float_Bounded('Factor: ', 0.1, 0, 1, 0.01)
        self.patience_w = ipwidget_basic.set_Int('Factor: ', 10, show = False)
        
        # parameters Cyclic LR
        self.cyclic_min_lr_w = ipwidget_basic.set_Float_Bounded('Min lr: ', 0.0001, 0, 1, 0.0001)
        self.cyclic_max_lr_w = ipwidget_basic.set_Float_Bounded('Max lr: ', 0.01, 0, 1, 0.01)
        
        # parameters cosine annealing
        self.cyclic_epochs_w = ipwidget_basic.set_Int('N epochs to restart LR: ', 50, show = False)
        
        # parameters step
        self.step_w = ipwidget_basic.set_Int('Step: ', 10, show = False)
        
        
        self.main_container = widgets.HBox(children= [self.factor_w, self.patience_w])
        
        display(self.main_container)
        
    def dropdown_handler_lr_scheduler(self, change):
        if change.new == 'reduce_on_plateau':
            self.main_container.children = [self.factor_w, self.patience_w]
        elif change.new == 'cyclic':
            self.main_container.children = [self.cyclic_min_lr_w, self.cyclic_max_lr_w, self.cyclic_epochs_w]
        elif change.new == 'cosine_annealing':            
            self.main_container.children = [self.cyclic_epochs_w]
        elif change.new == 'step':
            self.main_container.children = [self.step_w, self.factor_w]
    
    def get(self, optimizer):
        if self.optimizer_w.value == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=self.factor_w.value, patience=self.patience_w.value)
        elif self.optimizer_w.value == 'cyclic':
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode = 'min', base_lr = self.cyclic_min_lr_w.value, max_lr = self.cyclic_max_lr_w.value, step_size_up = self.cyclic_epochs_w.value)
        elif self.optimizer_w.value == 'cosine_annealing':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.cyclic_epochs_w.value)
        elif self.optimizer_w.value == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_w.value, gamma= self.factor_w.value)
        return lr_scheduler

################################################################################        
            
        
    