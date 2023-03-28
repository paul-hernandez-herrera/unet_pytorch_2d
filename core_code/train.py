import argparse 
import torch
from pathlib import Path
import copy

class train_model():
    def __init__(self, model,
                 train_dataloader,
                 validation_dataloader,
                 list_loss_function,
                 device,
                 optimizer,
                 epochs,
                 model_output_folder = '',
                 model_checkpoint = False,
                 model_checkpoint_frequency = 10):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.list_loss_function = list_loss_function
        self.device = device
        self.optimizer = optimizer
        self.epochs = epochs
        self.model_output_folder = model_output_folder
        self.model_checkpoint = model_checkpoint
        self.model_checkpoint_frequency = model_checkpoint_frequency

        
    def run(self):
        minimum_loss = float('inf')
        print('Running training')
        for epoch in range(1, self.epochs+1):
            print('Running iteration ' + str(epoch) + '/' +str(self.epochs))
            self.model.train() #set the model in training mode
            epoch_loss = 0
            for batch_iter in self.train_dataloader: #this in the main code to update weights      
                imgs, targets = batch_iter #getting imgs and target output for current batch
                
                #we have a tensor in the train_dataloader, move to device
                imgs = imgs.to(device=self.device, dtype = torch.float32)
                targets = targets.to(device=self.device, dtype=torch.float32)
                #targets = targets.to(device=self.device, dtype=torch.long)
                
                self.optimizer.zero_grad()  #Sets the gradients of all optimized torch.Tensor s to zero.
                
                network_output = self.model(imgs) #applying the model to the input images
                
                loss = self.list_loss_function[0](network_output, targets) #compute the error between the network output and target output
                for i in range(1,len(self.list_loss_function)):
                    loss += self.list_loss_function[i](network_output, targets) #compute the error between the network output and target output
                
                
                loss.backward() #compute the gradients given the loss value
                
                self.optimizer.step() #update the weights using the gradients and the approach by optimizer
                
                epoch_loss += loss.item()
            print(f"epoch loss: {epoch_loss}")
            if epoch_loss < minimum_loss:
                minimum_loss, epoch_best = epoch_loss, epoch
                model_best_state_dict = copy.deepcopy(self.model.state_dict())
                
            if self.model_checkpoint & ((epoch%self.model_checkpoint_frequency) ==0):
                torch.save(self.model.state_dict(), Path(self.model_output_folder, f"model_{epoch}.pth"))
                
        torch.save(self.model.state_dict(), Path(self.model_output_folder, "model_last.pth"))
        torch.save(model_best_state_dict, Path(self.model_output_folder, f"model_best_e{epoch_best}.pth"))
        return self.model
                
                
        
    
    
if __name__== '__main__':
    
    #to define
    parser = argparse.ArgumentParser(description='Main code to train the U-Net given a training set of 2D-input images and target image.')
    parser.add_argument('--folder_imgs', default=[], type=str, help='Path to the folder containing the input-images in the training set', required = True)
    parser.add_argument('--folder_target', default=[], type=str, help='Path to the folder containing the target-images in the training set.', required = True)
    args = parser.parse_args()