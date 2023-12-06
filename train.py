import torch
import torch.optim as optim
from models.cycle_gan import CycleGAN
from utils.data_loader import get_data_loader
import os
from torchvision import transforms 
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_dataloader, output_folder="output_images"):
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    
    for i, real_img in enumerate(test_dataloader):
        real_img = real_img.to(device)
        
        with torch.no_grad():
            fake_img = model.generator_SketchToReal(real_img)  

        real_pil = transforms.ToPILImage()(real_img[0].cpu())
        fake_pil = transforms.ToPILImage()(fake_img[0].cpu())
        
        real_path = os.path.join(output_folder, f"real_image_{i}.png")
        fake_path= os.path.join(output_folder, f"fake_image_{i}.png")
        
        real_pil.save(real_path)
        fake_pil.save(fake_path)
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(real_pil)
        plt.title(f"Real Image {i}")
        
        plt.subplot(1, 2, 2)
        plt.imshow(fake_pil)
        plt.title(f"Generated Image {i}")
        
        plt.savefig(os.path.join(output_folder, f"comparison_{i}.png"))
        plt.close()
        
    print(f"Evaluated Images saved in the '{output_folder}'")


def train(model, dataloader_A, dataloader_B, num_epochs=10, lr=0.0002, save_folder="saved_models"):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(save_folder, exist_ok=True)

    criterion_MSE = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for real_A, real_B in zip(dataloader_A, dataloader_B):
            real_A, real_B = real_A.to(device), real_B.to(device)

            fake_A, fake_B = model(real_A, real_B)
            
          
            pred_fake_A, pred_real_A = model.discriminator_A(fake_B, real_A)
            adv_loss_A2B = criterion_MSE(pred_fake_A[0], torch.zeros_like(pred_fake_A[0]))
            adv_loss_B2A = criterion_MSE(pred_real_A[0], torch.ones_like(pred_real_A[0]))
            
            
            resized_fake_A = torch.nn.functional.interpolate(fake_A, size=(real_A.size(2), real_A.size(3)), mode='bilinear', align_corners=False)
            cycle_loss_A = criterion_MSE(resized_fake_A, real_A)
            
           
            pred_fake_B, pred_real_B = model.discriminator_B(fake_A, real_B)
            adv_loss_B2A = criterion_MSE(pred_fake_B[0], torch.zeros_like(pred_fake_B[0]))
            adv_loss_A2B = criterion_MSE(pred_real_B[0], torch.ones_like(pred_real_B[0]))
            
            resized_fake_B = torch.nn.functional.interpolate(fake_B, size=(real_B.size(2), real_B.size(3)), mode='bilinear', align_corners=False)
            cycle_loss_B = criterion_MSE(resized_fake_B, real_B)

            total_gen_loss = adv_loss_A2B + adv_loss_B2A + cycle_loss_A + cycle_loss_B

            optimizer.zero_grad()
            total_gen_loss.backward()
            optimizer.step()

        save_path = os.path.join(save_folder, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)



data_root_A = "data/cars_train/cars_train"
data_root_B = "data/Sketches"
data_root_C = "data/cars_test/cars_test"
save_models_folder = "saved_models"

batch_size = 16

dataloader_A = get_data_loader(data_root_A, batch_size)
dataloader_B = get_data_loader(data_root_B, batch_size)
dataloader_C = get_data_loader(data_root_C, batch_size)

model = CycleGAN(in_channels=3, out_channels=3)
model.to(device)
train(model, dataloader_A, dataloader_B, num_epochs=10, lr=0.0002, save_folder=save_models_folder)
evaluate_model(model, dataloader_C)
