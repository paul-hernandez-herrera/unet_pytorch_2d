from core_code.loss.binary_loss import BinaryLoss
import numpy as np
import torch
import unittest

class binary_loss(unittest.TestCase):
    def test_binary_loss(self):
        np.random.seed(7)
        A = np.random.rand(20, 20) > 0.6
        GT = np.random.rand(20, 20) > 0.3
        
        TP = np.sum(A & GT)
        FP = np.sum(~A & GT)
        FN = np.sum(~GT & A)
        a, b, c = TP, FP, FN
        # Convert to shape (B, W, H)
        A = A[None]
        GT = GT[None]
        
        print(f"TP = {TP} --- FP = {FP} --- FN = {FN}")
        # Convert to tensors and convert A to logit because binary loss applies a sigmoid
        GT = torch.tensor(GT)
        A = torch.tensor((A * 2 - 1) * float('inf'))
        
        precision = a/(a+b)
        loss = BinaryLoss(option='precision')
        loss_val = 1-loss(A, GT)
        assert precision == loss_val
        print('Pass precision')
        
        recall = a/(a+c)
        loss = BinaryLoss(option='recall')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( recall, loss_val.numpy(), 7, "Failed recall")
        print('Pass recall')    
        
        kulczynski_I = a/(b+c)
        loss = BinaryLoss(option='kulczynski_I')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( kulczynski_I, loss_val.numpy(), 7, "Failed kulczynski_I")
        print('Pass kulczynski_I')      

        
        dice = 2*a/(2*a+b+c)
        loss = BinaryLoss(option='dice')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( dice, loss_val.numpy(), 7, "Failed dice")
        print('Pass dice')         
        
        
        sw_jaccard = 3*a/(3*a+b+c)
        loss = BinaryLoss(option='sw_jaccard')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( sw_jaccard, loss_val.numpy(), 7, "Failed sw_jaccard")
        print('Pass sw_jaccard')          
        
        
        jaccard = a/(a+b+c)
        loss = BinaryLoss(option='jaccard')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( jaccard, loss_val.numpy(), 7, "Failed jaccard")
        print('Pass jaccard')
        
        
        sokal_and_sneath_I = a/(a+2*b+ 2*c)
        loss = BinaryLoss(option='sokal_and_sneath_I')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( sokal_and_sneath_I, loss_val.numpy(), 7, "Failed sokal_and_sneath_I")
        print('Pass sokal_and_sneath_I') 
        
        
        Van_der_Maarel = (2*a-b-c)/(2*a+b+ c)
        loss = BinaryLoss(option='Van_der_Maarel')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( Van_der_Maarel, loss_val.numpy(), 7, "Failed Van_der_Maarel")
        print('Pass Van_der_Maarel')     
        
        
        johnson = a/(a+b) + a/(a+c)
        loss = BinaryLoss(option='johnson')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( johnson, loss_val.numpy(), 7, "Failed johnson")
        print('Pass johnson')  
        
        
        mcconaughey = (a*a - b*c)/((a+b)*(a+c))
        loss = BinaryLoss(option='mcconaughey')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( mcconaughey, loss_val.numpy(), 7, "Failed mcconaughey")
        print('Pass mcconaughey')         
        
                
        kulczynski_II = 0.5*(a/(a+b) + a/(a+c))
        loss = BinaryLoss(option='kulczynski_II')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( kulczynski_II, loss_val.numpy(), 7, "Failed kulczynski_II")
        print('Pass kulczynski_II')
        
        
        sorgenfrei = a*a/((a+b)*(a+c))
        loss = BinaryLoss(option='sorgenfrei')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( sorgenfrei, loss_val.numpy(), 7, "Failed sorgenfrei")
        print('Pass sorgenfrei')     
        
        
        driver_kroeber_ochiai = a/np.sqrt((a+b)*(a+c))
        loss = BinaryLoss(option='driver_kroeber_ochiai')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( driver_kroeber_ochiai, loss_val.numpy(), 7, "Failed driver_kroeber_ochiai")
        print('Pass driver_kroeber_ochiai') 
        
        
        braun_blanquet = a/np.maximum(a+b,a+c)
        loss = BinaryLoss(option='braun_blanquet')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( braun_blanquet, loss_val.numpy(), 7, "Failed braun_blanquet")
        print('Pass braun_blanquet')

        simpson = a/np.minimum(a+b,a+c)
        loss = BinaryLoss(option='simpson')
        loss_val = 1-loss(A, GT)
        self.assertAlmostEqual( simpson, loss_val.numpy(), 7, "Failed simpson")
        print('Pass simpson')    

        
if __name__=='__main__':
    unittest.main()
    
    