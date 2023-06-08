from . import UNet2d_core
from torch import nn

class Classic_UNet_2D(nn.Module):
    def __init__(self, input_n_channels, target_n_labels, mode_up = 'conv_transp'):
        super().__init__()
        self.input_n_channels = input_n_channels
        self.target_n_labels = target_n_labels
        
        self.encoder1 = UNet2d_core.Encoder_2D(self.input_n_channels, 64, 64) 
        self.encoder2 = UNet2d_core.Encoder_2D(64, 128, 128) 
        self.encoder3 = UNet2d_core.Encoder_2D(128, 256, 256)
        self.encoder4 = UNet2d_core.Encoder_2D(256, 512, 512)
        
        #''Bottom''
        self.encoder_bottom = UNet2d_core.Encoder_2D(512, 1024, 1024)
        
        self.decoder4 = UNet2d_core.Decoder_2D(512+512, 512, 512, upsampling_mode = mode_up)
        self.decoder3 = UNet2d_core.Decoder_2D(256+256, 256, 256, upsampling_mode = mode_up)
        self.decoder2 = UNet2d_core.Decoder_2D(128+128, 128, 128, upsampling_mode = mode_up)
        self.decoder1 = UNet2d_core.Decoder_2D(64+64, 64, 64, upsampling_mode = mode_up)
        
        self.outConv = nn.Conv2d( 64,  self.target_n_labels, kernel_size = (1,1))
        
        
    def forward(self, x):
        
        features_enconder1, x_down = self.encoder1(x)
        features_enconder2, x_down = self.encoder2(x_down)
        features_enconder3, x_down = self.encoder3(x_down)
        features_enconder4, x_down = self.encoder4(x_down)
        features_bottom, _ = self.encoder_bottom(x_down)
        x = self.decoder4(features_enconder4, features_bottom)
        x = self.decoder3(features_enconder3, x)
        x = self.decoder2(features_enconder2, x)
        x = self.decoder1(features_enconder1, x)
        x = self.outConv(x) 
        
        return x
        

    
    
    
        
        