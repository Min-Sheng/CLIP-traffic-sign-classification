
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


class TrafficSignDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row['Path']).convert('RGB')
        image = self.transform(image)

        label = row["ClassId"]

        # Use class name as prompt, other wise, fallback
        if 'ClassName' in row:
            classname = row['ClassName']
            text_prompt = f"A photo of a {classname} traffic sign"
        else:
            text_prompt = f"A traffic sign with class ID {row['ClassId']}"

        return image, label, text_prompt


def train_transform(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return transform(image)


def val_transform(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return transform(image)


def unnormalize(img_tensor):
    """Unnormalize a tensor image."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return img_tensor * std + mean
