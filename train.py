import torch
import torch.optim as optim
from models.cycle_gan import CycleGAN
from models.discriminator import Discriminator
from models.generator import Generator
from utils.data_loader import DataLoader
from utils.data_loader import get_data_loader
import os

def train(model, dataloader_A, dataloader_B, num_epochs=10, lr=0.0002, save_folder="saved_models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(save_folder, exist_ok=True)
    
    criterion_MSE = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for real_A, real_B in zip(dataloader_A, dataloader_B):
            real_A, real_B = real_A.to(device), real_B.to(device)

            
            fake_A, fake_B, reconstructed_A, reconstructed_B = model(real_A, real_B)

            pred_fake_A = model.discriminator_A(fake_A)
            pred_real_A = model.discriminator_A(real_A)
            pred_fake_B = model.discriminator_B(fake_B)
            pred_real_B = model.discriminator_B(real_B)

            adv_loss_A2B = criterion_MSE(pred_fake_B, torch.ones_like(pred_fake_B))
            adv_loss_B2A = criterion_MSE(pred_fake_A, torch.ones_like(pred_fake_A))
            
            cycle_loss_A = criterion_MSE(reconstructed_A, real_A)
            cycle_loss_B = criterion_MSE(reconstructed_B, real_B)

            total_gen_loss = adv_loss_A2B + adv_loss_B2A + cycle_loss_A + cycle_loss_B

            optimizer.zero_grad()
            total_gen_loss.backward()
            optimizer.step()

            
            save_path = os.path.join(save_folder, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
         
data_root_A = "data/cars_train/cars_train"
data_root_B = "data/Sketches"
batch_size = 16

dataloader_A = get_data_loader(data_root_A, batch_size)
dataloader_B = get_data_loader(data_root_B, batch_size)

save_models_folder = "saved_models"

model = CycleGAN()
train(model, dataloader_A, dataloader_B, num_epochs=10, lr=0.0002, save_folder=save_models_folder)