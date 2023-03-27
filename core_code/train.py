import argparse 
import torch 

class train_model():
    def __init__(self, model,
                 train_dataloader,
                 validation_dataloader,
                 list_loss_function,
                 device,
                 optimizer,
                 epochs):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.list_loss_function = list_loss_function
        self.device = device
        self.optimizer = optimizer
        self.epochs = epochs

        
    def run(self):        
        print('Running training')
        for epoch in range(0, self.epochs):
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
            epoch_loss = epoch_loss/len(self.train_dataloader.dataset)
            print(epoch_loss)
        return self.model
                
                
        
    
    
if __name__== '__main__':
    
    #to define
    parser = argparse.ArgumentParser(description='Main code to train the U-Net given a training set of 2D-input images and target image.')
    parser.add_argument('--folder_imgs', default=[], type=str, help='Path to the folder containing the input-images in the training set', required = True)
    parser.add_argument('--folder_target', default=[], type=str, help='Path to the folder containing the target-images in the training set.', required = True)
    args = parser.parse_args()