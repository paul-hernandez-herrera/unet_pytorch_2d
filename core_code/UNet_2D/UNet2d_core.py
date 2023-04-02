from torch import cat, nn
from torchvision.transforms.functional import center_crop
##########################################

class SingleConv2D(nn.Module):
    def __init__(self, in_channels, num_filters_conv):
        super().__init__()
        self.seq_single_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,  out_channels = num_filters_conv, kernel_size=3, padding = 1),
            nn.BatchNorm2d(num_filters_conv),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.seq_single_conv(x)            

##########################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, num_filters_conv1, num_filters_conv2):
        super().__init__()
        self.double_conv = nn.Sequential(
            SingleConv2D(in_channels, num_filters_conv1),
            SingleConv2D(num_filters_conv1, num_filters_conv2)
        )
        
    def forward(self, x):
        return self.double_conv(x)

##########################################

class UpSampling(nn.Module):
    #This function upsample the image/features size (width, height) and reduces the number of channels by half
    def __init__(self,  in_channels, mode = 'conv_transp'):
        super().__init__()
        
        if mode == 'conv_transp':
            self.upsampling = nn.ConvTranspose2d(in_channels = in_channels, out_channels = in_channels//2, kernel_size=2, stride=2)
        elif mode == 'bilinear':
            self.upsampling = nn.sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                SingleConv2D(in_channels,in_channels//2)
            )
        else:
            raise ValueError('Upsampling method not recognized')        
    
    def forward(self, x):
        return self.upsampling(x)
            
##########################################

def concatenate_tensors(x1, x2):
    # for a 2d image shape is (B,C,W,H)
    #there can be mismatch between the size of the images to concatenate. This function allows to fix this mismatch
    x2 = center_crop(x2,[x1.shape[2], x1.shape[3]])
    
    return cat((x1, x2), dim=1)

##########################################

class Encoder_2D(nn.Module):
    def __init__(self, in_channels, num_filters_conv1, num_filters_conv2):
        super().__init__()
        if num_filters_conv2 % 2 != 0:
            raise ValueError('Encoder can not have odd number of filters (channels) in the second convolutional layer.')
            
        self.encoder = DoubleConv(in_channels, num_filters_conv1, num_filters_conv2)
        self.pool  = nn.MaxPool2d(kernel_size = (2,2))
    
    def forward(self, x):
        x = self.encoder(x)
        x_down = self.pool(x)
        return x, x_down
    
##########################################

class Decoder_2D(nn.Module):
    def __init__(self, in_channels, num_filters_conv1, num_filters_conv2, upsampling_mode = 'conv_transp'):
        super().__init__()
        self.doubleConv =  DoubleConv(in_channels, num_filters_conv1, num_filters_conv2)
        self.upsampling = UpSampling(in_channels, mode = upsampling_mode)
    
    def forward(self, x_decoder, x):
        x = self.upsampling(x)
        x = concatenate_tensors(x_decoder, x)
        
        return self.doubleConv(x)
        
        
        
        
        
        