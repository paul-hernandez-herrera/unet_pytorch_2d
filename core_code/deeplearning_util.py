import torch

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

def compute_validation_loss(model, validation_dataloader, list_loss_functions, device):
    
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