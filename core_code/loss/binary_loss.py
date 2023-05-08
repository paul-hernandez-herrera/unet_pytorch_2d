import torch

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, model_output, target):
        # this function computer the dice loss function for two tensor. 
        # It assumes that model_output and target are of the same size [B, C, W, H]
        # We assume that the model returns the is not normalized to probabilities [0,1].
        
        # Normalizing to [0,1]
        output_normalized_0_1 = torch.sigmoid(model_output)
        
        # convert to 1-d vector
        output_normalized_0_1 = output_normalized_0_1.view(-1)
        target = target.view(-1)
        
        # calculating the metrics over vectors in general terms
        A_int_B = (output_normalized_0_1 * target).sum() 
        A = output_normalized_0_1.sum() 
        B = target.sum()
        
        if A+B==0:
            dice = 1
        else:
            dice = 2*A_int_B/(A+B)
        
        # goal minimize the metric. Dice best performance is at maximum value equal to one, then substracting one
        return 1-dice