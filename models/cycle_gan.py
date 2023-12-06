import torch
import torch.nn as nn
from .generator import Generator  
from .discriminator import Discriminator 
import torch.optim as optim

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CycleGAN, self).__init__()

        self.generator_SketchToReal = Generator(in_channels, out_channels)
        self.generator_RealToSketch = Generator(out_channels, in_channels)  

        self.discriminator_Sketch = Discriminator(in_channels)
        self.discriminator_Real = Discriminator(in_channels)  

    def forward(self, x_Sketch, x_Real):
        fake_Real = self.generator_SketchToReal(x_Sketch)
        fake_Sketch = self.generator_RealToSketch(x_Real)
        return fake_Real, fake_Sketch

    def discriminator_A(self, x_sketch, x_real):
        pred_fake_Sketch = self.discriminator_Sketch(x_sketch)
        pred_real_A = self.discriminator_Real(x_real)
        return pred_fake_Sketch, pred_real_A

    def discriminator_B(self, x_sketch, x_real):
        pred_fake_B = self.discriminator_Sketch(x_sketch)
        pred_real_B = self.discriminator_Real(x_real)
        return pred_fake_B, pred_real_B
