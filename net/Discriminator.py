import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# 定义 BasicBlock，添加 Spectral Normalization
class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True, use_spectral_norm=False):
        super().__init__()
        # 添加 Spectral Normalization
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)
        
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        fx = self.conv(x)
        if self.isn is not None:
            fx = self.isn(fx)
        fx = self.lrelu(fx)
        return fx

# 定义 ConditionalDiscriminator，添加 Spectral Normalization
class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator"""
    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(2, 64, norm=False, use_spectral_norm=True)  # 64 * 128 * 128
        self.block2 = BasicBlock(64, 128, use_spectral_norm=True)           # 128 * 64 * 64
        self.block3 = BasicBlock(128, 256, use_spectral_norm=True)          # 256 * 32 * 32
        self.block4 = spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1))  # 1 * 32 * 32# 对最后一层添加 Spectral Normalization
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层

    def forward(self, x, cond):
        """
        Args:
            x: 输入图像张量
            cond: 条件张量
        """
        x = torch.cat([x, cond], dim=1)  # 拼接输入和条件
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.dropout(fx)  # 添加 Dropout
        return fx

class ResidualBasicBlock(nn.Module):
    """Basic block with residual connection"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True, use_spectral_norm=False):
        super().__init__()
        
        # Main convolutional path
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)
        
        # Optional normalization
        self.isn = nn.InstanceNorm2d(outplanes) if norm else None
        
        # Activation
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # Residual connection
        if stride != 1 or inplanes != outplanes:
            self.residual = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0)
            if use_spectral_norm:
                self.residual = spectral_norm(self.residual)
        else:
            self.residual = nn.Identity()
        
    def forward(self, x):
        # Main path
        fx = self.conv(x)
        if self.isn is not None:
            fx = self.isn(fx)
        fx = self.lrelu(fx)
        
        # Residual path
        residual = self.residual(x)
        
        # Combine
        out = fx + residual
        return out

class ConditionalDiscriminatorRes(nn.Module):
    """Conditional Discriminator with Residual Connections"""
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBasicBlock(2, 64, norm=False, use_spectral_norm=True)   # Output: 64 x 128 x 128
        self.block2 = ResidualBasicBlock(64, 128, use_spectral_norm=True)            # Output: 128 x 64 x 64
        self.block3 = ResidualBasicBlock(128, 256, use_spectral_norm=True)           # Output: 256 x 32 x 32
        self.block4 = spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1))  # Output: 1 x 32 x 32
#        self.dropout = nn.Dropout(0.5)  # Dropout layer
    
    def forward(self, x, cond):
        """
        Args:
            x: Input image tensor
            cond: Conditional tensor
        """
        x = torch.cat([x, cond], dim=1)  # Concatenate input and condition along channel dimension
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
#        fx = self.dropout(fx)  # Apply Dropout
        return fx