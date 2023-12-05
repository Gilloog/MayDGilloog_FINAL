import torch
import torch.optim as optim
from models.cycle_gan import CycleGAN
from models.discriminator import Discriminator
from models.generator import Generator
from utils.data_loader import DataLoader
from utils.data_loader import get_data_loader

def train(model, dataloader_A, dataloader_B, num_epochs=10, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for real_A, real_B in zip(dataloader_A, dataloader_B):
            real_A, real_B = real_A.to(device), real_B.to(device)

            
            fake_A, fake_B, reconstructed_A, reconstructed_B = model(real_A, real_B)

            optimizer.zero_grad()
         
data_root_A = "data/cars_train"
data_root_B = "data/Sketches"
batch_size = 16

dataloader_A = get_data_loader(data_root_A, batch_size)
dataloader_B = get_data_loader(data_root_B, batch_size)

model = CycleGAN()
train(model, dataloader_A, dataloader_B)