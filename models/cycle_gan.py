import torch
import torch.nn as nn
from .generator import Generator
from .discriminator import Discriminator

class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.generator_A2B = Generator()
        self.generator_B2A = Generator()
        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()

    def forward(self, real_A, real_B):
        
        fake_B = self.generator_A2B(real_A)
        reconstructed_A = self.generator_B2A(fake_B)

        fake_A = self.generator_B2A(real_B)
        reconstructed_B = self.generator_A2B(fake_A)

        pred_fake_A = self.discriminator_A(fake_A)
        pred_real_A = self.discriminator_A(real_A)
        pred_fake_B = self.discriminator_B(fake_B)
        pred_real_B = self.discriminator_B(real_B)

        criterion_MSE = nn.MSELoss()

        adv_loss_A2B = criterion_MSE(pred_fake_B, torch.ones_like(pred_fake_B))
        adv_loss_B2A = criterion_MSE(pred_fake_A, torch.ones_like(pred_fake_A))

        cycle_loss_A = criterion_MSE(reconstructed_A, real_A)
        cycle_loss_B = criterion_MSE(reconstructed_B, real_B)

        total_gen_loss = adv_loss_A2B + adv_loss_B2A + cycle_loss_A + cycle_loss_B

        return fake_A, fake_B, reconstructed_A, reconstructed_B, total_gen_loss