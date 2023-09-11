import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        
        # hidden_channels 
        
        self.enc1 = self.conv_block(in_channels, 128)
        self.down1 = self.downconv_block(128, 256,True)
        self.enc2 = self.conv_block(256, 256)
        self.down2 = self.downconv_block(256, 512)
        self.enc3 = self.conv_block(512, 512)
        self.down3 = self.downconv_block(512, 1024)
        self.enc4 = self.conv_block(1024, 1024)
        
        
        # MaxPooling
        # self.pool = nn.MaxPool2d(2)
        
        # Decoder (Upsampling)
        self.up3 = self.upconv_block(1024, 512)
        self.dec3 = self.conv_block(1024, 512)
        self.up2 = self.upconv_block(512, 256)
        self.dec2 = self.conv_block(512, 256)
        self.up1 = self.upconv_block(256, 128,True)
        self.dec1 = self.conv_block(256, 128)
        
        # Final Output
        self.out = nn.Conv2d(128, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        enc4 = self.enc4(self.down3(enc3))
        
        # Decoder
        up3 = self.up3(enc4)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(merge3)
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(merge2)
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(merge1)
        
        # Output
        out = self.out(dec1)
        return out
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def downconv_block(self, in_channels, out_channels,first=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3+int(first==True),stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels,first=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2+(first==True), stride=2),
            nn.ReLU(inplace=True)
        )

if __name__ == '__main__':

    # Create a U-Net model instance
    unet_model = UNet(70, 5*10)  # Assuming 70-channel input and binary segmentation (1 output channel)

    # Print the U-Net model architecture
    print(unet_model)
    
    num_parameters = sum(p.numel() for p in unet_model.parameters())
    
    # Convert to megabytes
    num_megabytes = num_parameters * 4 / (1024 ** 2)
    
    print(f"Model has {num_parameters} parameters.")
    print(f"Model size is approximately {num_megabytes:.2f} MB.")
    
    


