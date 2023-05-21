import torch

class BinaryLoss(torch.nn.Module):
    def __init__(self, option = 'dice'):
        self.option = 'dice'
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
        
        if self.option == 'dice':
            # conditional probability --- 1/( 0.5*(1/P(A|B)) + 0.5*(1/P(B|A)))
            if A+B==0:
                loss = 1
            else:
                loss = 2*A_int_B/(A+B)
        elif self.option == 'jaccard':
            # conditional probability --- P(A^B|AUB)
            A_union_B = A + B + A_int_B 
            if A_union_B==0:
                loss = 1
            else:
                loss = A_int_B/A_union_B
        elif self.option == 'Sorgenfrei': 
            # conditional probability --- P(A|B)*P(B|A)
            if A==0 or B==0:
                loss = 1
            else:
                loss = (A_int_B/B)*(A_int_B/A)
        elif self.option == 'precision':
            # conditional probability --- P(A|B)
            if B==0:
                loss = 1
            else:
                loss = (A_int_B/B)
        elif self.option == 'recall':
            # conditional probability --- P(A|B)
            if A==0:
                loss = 1
            else:
                loss = (A_int_B/A)
            
        
        # goal minimize the metric. Dice best performance is at maximum value equal to one, then substracting one
        return 1-loss   
       