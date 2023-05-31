import torch

class BinaryLoss(torch.nn.Module):
    def __init__(self, option = 'dice'):
        self.option = option
        super().__init__()
        
    def forward(self, model_output, target):
        # this function computer the dice loss function for two tensor. 
        # It assumes that model_output and target are of the same size [B, C, W, H]
        # We assume that the model returns the is not normalized to probabilities [0,1].
        
        # Normalizing to [0,1]
        output = torch.sigmoid(model_output)
        
        # convert to 1-d vector
        output = output.view(-1)
        target = target.view(-1)
               
        if self.option == 'recall':
            # conditional probability --- P(ground_true|prediction)
            loss = general_equation(output, target, 0, 1, 1)
        elif self.option == 'precision':
            # conditional probability --- P(prediction|ground_true)
            loss = general_equation(output, target, 1, 0, 1)            
        elif self.option == 'kulczynski_I':
            loss = general_equation(output, target, 1, 1, 0)            
        elif self.option == 'dice':
            loss = general_equation(output, target, 1/2, 1/2, 1)
        elif self.option == 'sw_jaccard':
            loss = general_equation(output, target, 1/3, 1/3, 1)
        elif self.option == 'jaccard':
            loss = general_equation(output, target, 1  , 1  , 1)            
        elif self.option == 'sokal_and_sneath_I':
            loss = general_equation(output, target, 2  , 2  , 1)
        elif self.option == 'Van_der_Maarel':
            loss = 2*general_equation(output, target, 1/2, 1/2, 1) -1            
        elif self.option == 'johnson':
            loss = general_equation(output, target, 1, 0, 1) + general_equation(output, target, 0, 1, 1)
        elif self.option == 'mcconaughey':
            loss = general_equation(output, target, 1, 0, 1) + general_equation(output, target, 0, 1, 1) -1
        elif self.option == 'kulczynski_II':
            loss = (general_equation(output, target, 1, 0, 1) + general_equation(output, target, 0, 1, 1) ) / 2
        elif self.option == 'sorgenfrei':
            loss = general_equation(output, target, 1, 0, 1) * general_equation(output, target, 0, 1, 1)
        elif self.option == 'driver_kroeber_ochiai':
            loss = (general_equation(output, target, 1, 0, 1) * general_equation(output, target, 0, 1, 1)).sqrt()
        elif self.option == 'braun_blanquet':
            loss = torch.minimum(general_equation(output, target, 1, 0, 1), general_equation(output, target, 0, 1, 1))
        elif self.option == 'simpson':
            loss = torch.maximum(general_equation(output, target, 1, 0, 1), general_equation(output, target, 0, 1, 1))
            
        # goal minimize the metric. Dice best performance is at maximum value equal to one, then substracting one
        return 1-loss  

def intersection(A, B):
    return A * B

def union(A, B):
    return A+B-intersection(A, B)
    
def conditional_probability(A, B):
    cardinality_A_int_B = (A*B).sum()
    cardinality_B = B.sum()
    if cardinality_B == 0:
        return 0
    else:
        return cardinality_A_int_B/cardinality_B
    
def general_equation(A, B, alpha, beta, gamma):
    if [alpha,beta,gamma]==[1,0,1]:
        return conditional_probability(A, B)
    elif [alpha,beta,gamma]==[0,1,1]:
        return conditional_probability(B, A)        
    else:
        Prob_A_B = conditional_probability(A, B)
        Prob_B_A = conditional_probability(B, A)        
        return 1/((alpha*((1/Prob_A_B)-1) ) + (beta*((1/Prob_B_A)-1) ) + gamma)
    
       