from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images[index])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
def get_data_loader(root, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader