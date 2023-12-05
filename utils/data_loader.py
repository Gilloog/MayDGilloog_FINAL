from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def init(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)

    def len(self):
        return len(self.images)

    def getitem(self, index):
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